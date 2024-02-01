import numpy as np
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedVideoSampler(_DistributedSampler):
    """Put videos to multi gpus during testing.

    Args:
        dataset (Dataset): Test dataset that must has `data_infos` attribute.
            Each data_info in `data_infos` record information of one frame,
            and each video must has one data_info that includes
            `data_info['frame_id'] == 0`.
        num_replicas (int): The number of gpus. Defaults to None.
        rank (int): Gpu rank id. Defaults to None.
        shuffle (bool): If True, shuffle the dataset. Defaults to False.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        assert not self.shuffle, "Specific for video sequential testing."
        self.num_samples = len(dataset)

        first_frame_indices = []
        for i, img_info in enumerate(self.dataset.data_infos):
            if img_info["frame_id"] == 0:
                first_frame_indices.append(i)

        if len(first_frame_indices) < num_replicas:
            raise ValueError(
                f"only {len(first_frame_indices)} videos loaded,"
                f"but {self.num_replicas} gpus were given."
            )

        chunks = np.array_split(first_frame_indices, self.num_replicas)
        split_flags = [c[0] for c in chunks]
        split_flags.append(self.num_samples)

        rank, _ = get_dist_info()

        # split wrt #videos
        for i in range(len(split_flags) - 1):
            if rank == 0:
                print("split dataset wrt #videos:")
                print(
                    "rank[{}] num frames: {}".format(
                        i, split_flags[i + 1] - split_flags[i]
                    )
                )

        num_frames_per_gpu = len(self.dataset) // self.num_replicas
        split_flags = [0]
        _rank = 0
        for _frame in first_frame_indices:
            if _frame - split_flags[_rank] > num_frames_per_gpu:
                split_flags.append(_frame)
                _rank += 1

            if len(split_flags) == self.num_replicas:
                break
        split_flags.append(self.num_samples)

        # split wrt #frames
        for i in range(len(split_flags) - 1):
            if rank == 0:
                print("split dataset wrt #frames:")
                print(
                    "rank[{}] num frames: {}".format(
                        i, split_flags[i + 1] - split_flags[i]
                    )
                )

        self.indices = [
            list(range(split_flags[i], split_flags[i + 1]))
            for i in range(self.num_replicas)
        ]

        # # debug
        # skip first 2390 iters
        # skip = 2390
        # self.indices = [self.indices[i][skip:]
        #                 for i in range(self.num_replicas)
        #                 ]
        #
        # rank2indices = {
        #     0: 0,
        #     1: 32742,
        #     2: 67251,
        #     3: 122098
        # }
        # rank2Bcheck = 0
        # print("Check rank{}".format(rank2Bcheck))
        # self.indices[0] = self.indices[0][rank2indices[rank2Bcheck]:]
        # print("Datasampler: rank {}\t indices{}".format(rank2Bcheck, self.indices[self.rank][:10]))

    def __iter__(self):
        """Put videos to specify gpu."""
        indices = self.indices[self.rank]
        return iter(indices)

import torch
from mmcv.runner import BaseModule
# from ..aggregators.selsa_aggregator import SelsaAggregator


class MemoryBank(BaseModule):
    def __init__(self,
                 max_length=20000, key_length=2000,
                 sampling_policy='random', updating_policy='random',
                 ):
        super().__init__()
        self.max_length = max_length
        self.key_length = key_length
        self.sampling_policy = sampling_policy
        self.updating_policy = updating_policy
        self.feat = None
        # self.aggregator = SelsaAggregator(in_channels)

    def reset(self):
        self.feat = None

    def init_memory(self, feat):
        """
        init memory
        Args:
            feat: tensor [n, c]

        Returns:

        """
        self.feat = feat

    def sample(self):
        if self.feat is None:
            # write first
            return []

        if len(self.feat) < self.key_length:
            return self.feat

        if self.sampling_policy == 'random':
            sampled_ind = torch.randperm(len(self.feat))[:self.key_length]
            return self.feat[sampled_ind]
        else:
            raise NotImplementedError

    def update(self, new_feat):
        if self.feat is None:
            # first time
            self.feat = new_feat
            return

        if len(self.feat) < self.max_length:
            self.feat = torch.cat([self.feat, new_feat], dim=0)
            return

        if self.updating_policy == "random":
            new_num = len(new_feat)
            reserved_ind = torch.randperm(len(self.feat))[:-new_num]
            self.feat = torch.cat([self.feat[reserved_ind], new_feat], dim=0)
        else:
            raise NotImplementedError("not implemented")

    def __len__(self):
        if self.feat is None:
            return 0
        return len(self.feat)

    # def forward(self, x, x_support=None):
    #     # inference
    #     if x_support is None:
    #         raise NotImplementedError
    #     # training
    #     else:
    #         x = x + self.aggregator(x, x_support)
    #         return x

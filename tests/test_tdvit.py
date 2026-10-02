"""TDViT: the temporal dilated transformer block, its memory, the backbone's
training and online paths, the detector around it, and the data it trains on
(one reference per stage, crops shared by a clip)."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from vfe.config import Config
from vfe.datasets.imagenet_vid import ImagenetVIDDataset
from vfe.datasets.pipelines import SeqRandomCrop
from vfe.models.backbones import SwinTransformer, TDViT
from vfe.models.backbones.swin import SwinBlock
from vfe.models.backbones.tdvit import TDTB, MemoryQueue
from vfe.models.builder import build_model
from vfe.models.vid.tdvit import TDViTDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs/vid/tdvit"
TINY = dict(embed_dims=32, num_heads=[1, 2, 4, 8], window_size=7, mlp_ratio=4,
            drop_path_rate=0.0)


def tiny_tdvit(**kwargs) -> TDViT:
    torch.manual_seed(0)
    model = TDViT(**{**TINY, **kwargs})
    model.init_weights()
    for name, p in model.named_parameters():
        if "relative_position_bias_table" in name:
            torch.nn.init.normal_(p, std=0.5)  # zeros by default: make the bias matter
    return model


def block(cls=TDTB, shift=False, window_size=4, **kwargs):
    torch.manual_seed(0)
    b = cls(embed_dims=16, num_heads=2, feedforward_channels=32, window_size=window_size,
            shift=shift, **kwargs)
    torch.nn.init.normal_(b.attn.w_msa.relative_position_bias_table, std=0.5)
    return b.eval()


# ---- the block --------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [False, True])
def test_a_tdtb_attending_to_its_own_frame_is_a_swin_block(shift):
    # Pins the q/k/v split of the shared projection, the windows, the shift
    # mask and the position bias of the cross-attention path.
    tdtb = block(shift=shift)
    swin = block(SwinBlock, shift=shift)
    swin.load_state_dict(tdtb.state_dict())
    x = torch.randn(2, 10 * 11, 16)  # padded to 12 x 12: four windows
    with torch.no_grad():
        expected = swin(x, (10, 11))
        assert torch.equal(tdtb(x, (10, 11)), expected)  # no reference: Swin's own path
        torch.testing.assert_close(tdtb(x, (10, 11), ref=x), expected, rtol=1e-5, atol=1e-5)
        assert not torch.allclose(tdtb(x, (10, 11), ref=torch.randn_like(x)), expected)


def test_a_query_sees_only_the_reference_tokens_of_its_window():
    tdtb = block()
    x, ref = torch.randn(1, 8 * 8, 16), torch.randn(1, 8 * 8, 16)
    changed = ref.clone().view(1, 8, 8, 16)
    changed[:, 4:, 4:] += torch.randn(1, 4, 4, 16)  # the bottom-right 4 x 4 window only
    with torch.no_grad():
        a = tdtb(x, (8, 8), ref=ref).view(1, 8, 8, 16)
        b = tdtb(x, (8, 8), ref=changed.view(1, 64, 16)).view(1, 8, 8, 16)
    assert torch.equal(a[:, :4], b[:, :4]) and torch.equal(a[:, 4:, :4], b[:, 4:, :4])
    assert not torch.allclose(a[:, 4:, 4:], b[:, 4:, 4:])


def test_the_reference_has_no_gradient_but_the_projection_does():
    tdtb = block().train()
    x = torch.randn(1, 64, 16, requires_grad=True)
    ref = torch.randn(1, 64, 16, requires_grad=True)
    tdtb(x, (8, 8), ref=ref.detach()).sum().backward()
    qkv = tdtb.attn.w_msa.qkv.weight.grad
    assert qkv[16:].abs().sum() > 0  # keys and values are learnt from the reference
    assert ref.grad is None


# ---- the memory -------------------------------------------------------------------------


def frames(n, b=1, hw=(4, 6), c=3):
    return [torch.full((b, hw[0] * hw[1], c), float(i)) for i in range(n)]


def test_the_memory_returns_the_frame_itself_until_it_holds_one():
    memory = MemoryQueue(3)
    x = torch.randn(1, 24, 3)
    assert memory.sample(x, (4, 6)) is x and memory.reference is None


def test_earliest_draws_the_oldest_map_and_reuses_it_for_d_frames():
    d = 3
    memory = MemoryQueue(d)
    seq = frames(12)
    used = []
    for x in seq:
        used.append(int(memory.sample(x, (4, 6))[0, 0, 0]))
        memory.update(x)
    # Frame 0 attends to itself; draws at frames 1, 1 + d, 1 + 2d, ... take the
    # oldest of the last d frames, which then serves d frames.
    assert used == [0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 7, 7]
    assert len(memory) == d
    version = memory.version
    memory.reset()
    assert len(memory) == 0 and memory.reference is None and memory.version > version


def test_reuse_one_takes_the_frame_exactly_d_back():
    memory = MemoryQueue(4, reuse=1)
    used = []
    for x in frames(8):
        used.append(int(memory.sample(x, (4, 6))[0, 0, 0]))
        memory.update(x)
    assert used == [0, 0, 0, 0, 0, 1, 2, 3]


def test_temporal_nms_draws_the_map_with_the_largest_norm():
    memory = MemoryQueue(3, policy="nms")
    for value in (1.0, 5.0, -2.0):
        memory.update(torch.full((1, 24, 3), value))
    assert memory.sample(torch.zeros(1, 24, 3), (4, 6))[0, 0, 0] == 5.0


def test_channel_shuffle_takes_each_channel_from_some_frame():
    memory = MemoryQueue(4, policy="channel_shuffle")
    for x in frames(4, c=64):
        memory.update(x)
    torch.manual_seed(0)
    ref = memory.sample(torch.zeros(1, 24, 64), (4, 6))
    per_channel = ref[0].unique(dim=0)
    assert per_channel.shape[0] == 1  # one frame per channel, the same at every token
    assert set(per_channel[0].tolist()) == {0.0, 1.0, 2.0, 3.0}  # 64 channels hit every frame


def test_patch_shuffle_builds_the_quarters_from_four_groups_of_frames():
    memory = MemoryQueue(8, policy="patch_shuffle")
    for x in frames(8):
        memory.update(x)
    torch.manual_seed(0)
    ref = memory.sample(torch.zeros(1, 24, 3), (4, 6)).view(4, 6, 3)[..., 0]
    quarters = [ref[:2, :3], ref[:2, 3:], ref[2:, :3], ref[2:, 3:]]
    for q, (lo, hi) in zip(quarters, [(0, 1), (2, 3), (4, 5), (6, 7)], strict=True):
        assert q.unique().numel() == 1 and lo <= q[0, 0] <= hi  # group q's frames


# ---- the backbone -----------------------------------------------------------------------


def test_tdvit_t_has_swin_t_parameters_and_the_split_layout():
    tdvit = TDViT()
    swin = SwinTransformer()
    assert {k: v.shape for k, v in tdvit.state_dict().items()} == \
        {k: v.shape for k, v in swin.state_dict().items()}
    assert [s.layout for s in tdvit.stages] == ["st", "st", "sssttt", "st"]
    dilations = [[b.temporal_dilation for b in s.blocks if isinstance(b, TDTB)]
                 for s in tdvit.stages]
    assert dilations == [[4], [8], [16, 16, 16], [32]]
    # Shifts alternate with the index in the stage, as in Swin.
    assert [b.attn.shift_size > 0 for b in tdvit.stages[2].blocks] == [False, True] * 3


def test_the_advanced_variant_appends_tdtbs_without_stochastic_depth():
    tdvit = TDViT(extra_tdtbs=(0, 0, 2, 0), drop_path_rate=0.2)
    stage = tdvit.stages[2]
    assert stage.layout == "sssttttt"
    rates = [b.attn.drop.drop_prob if hasattr(b.attn.drop, "drop_prob") else 0.0
             for b in stage.blocks]
    assert rates[6:] == [0.0, 0.0] and rates[5] > 0
    swin_keys = set(SwinTransformer().state_dict())
    extra = set(tdvit.state_dict()) - swin_keys
    assert extra and all(k.startswith(("stages.2.blocks.6.", "stages.2.blocks.7.")) for k in extra)


def test_a_videos_first_frame_is_swin_and_later_ones_read_the_memory():
    tdvit = tiny_tdvit().eval()
    swin = SwinTransformer(depths=[2, 2, 6, 2], **TINY).eval()
    swin.load_state_dict(tdvit.state_dict())
    video = torch.randn(3, 1, 3, 72, 104)
    with torch.no_grad():
        first = tdvit(video[0])
        assert all(torch.equal(a, b) for a, b in zip(first, swin(video[0]), strict=True))
        second = tdvit(video[1])
        assert not all(torch.allclose(a, b) for a, b in zip(second, swin(video[1]), strict=True))
        lengths = [len(b.memory) for s in tdvit.stages for b in s.blocks if isinstance(b, TDTB)]
        assert lengths == [2] * 6
        tdvit.reset_memory()
        again = tdvit(video[0])  # a new video starts afresh
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))


@pytest.mark.parametrize("policy", ["earliest", "nms"])
def test_online_frames_attend_to_the_sampled_reference_with_cached_keys(policy):
    tdtb = block(shift=True, temporal_dilation=3, memory_sampling=policy)
    video = [torch.randn(1, 10 * 11, 16) for _ in range(9)]
    shadow = MemoryQueue(3, policy=policy)  # replays the schedule independently
    with torch.no_grad():
        for x in video:
            ref = shadow.sample(x, (10, 11))
            shadow.update(x)
            expected = tdtb(x, (10, 11)) if ref is x else tdtb(x, (10, 11), ref=ref)
            torch.testing.assert_close(tdtb.forward_online(x, (10, 11)), expected,
                                       rtol=1e-5, atol=1e-5)
    assert tdtb._kv_cache is not None  # keys and values were reused, not recomputed


def test_memory_keeps_the_output_when_asked():
    tdtb = block(temporal_dilation=2, memory_feature="output")
    x = torch.randn(1, 64, 16)
    with torch.no_grad():
        out = tdtb.forward_online(x, (8, 8))
    assert torch.equal(tdtb.memory.frames[-1], out)


@pytest.mark.parametrize("memory_feature", ["input", "output"])
def test_training_references_are_each_stage_frame_through_its_stages(memory_feature):
    tdvit = tiny_tdvit(memory_feature=memory_feature).eval()
    refs = torch.randn(2, 4, 3, 72, 104)
    with torch.no_grad():
        maps = tdvit.reference_maps(refs)
    # Expected: reference s of each clip alone through the network as a first
    # frame (every TDTB attending to itself), capturing what its stage-s TDTBs
    # receive (or return).
    seen = {}

    def capture(name):
        def hook(module, args, output):
            seen[name] = args[0] if memory_feature == "input" else output
        return hook

    handles = [b.register_forward_hook(capture((i, j)))
               for i, s in enumerate(tdvit.stages) for j, b in enumerate(s.blocks)
               if isinstance(b, TDTB)]
    for clip in range(2):
        for stage in range(4):
            seen.clear()
            tdvit.reset_memory()
            with torch.no_grad():
                tdvit(refs[clip, stage][None])
            for j, kept in enumerate(maps[stage]):
                if isinstance(tdvit.stages[stage].blocks[j], TDTB):
                    torch.testing.assert_close(kept[clip:clip + 1], seen[(stage, j)],
                                               rtol=1e-5, atol=1e-5)
                else:
                    assert kept is None
    for h in handles:
        h.remove()


def test_training_needs_references_and_reaches_every_parameter():
    tdvit = tiny_tdvit().train()
    with pytest.raises(ValueError, match="reference"):
        tdvit(torch.randn(1, 3, 72, 104))
    refs = torch.randn(2, 4, 3, 72, 104, requires_grad=True)
    outs = tdvit(torch.randn(2, 3, 72, 104), refs)
    assert [tuple(o.shape[:2]) for o in outs] == [(2, 32), (2, 64), (2, 128), (2, 256)]
    sum(o.square().mean() for o in outs).backward()
    assert all(p.grad is not None for p in tdvit.parameters())
    assert refs.grad is None  # references pass without gradients
    with pytest.raises(ValueError, match="one reference per stage"):
        tdvit(torch.randn(1, 3, 72, 104), torch.randn(1, 2, 3, 72, 104))


# ---- the detector -----------------------------------------------------------------------


def tiny_config(name="tdvit_t_frcnn_fpn_3x.py") -> dict:
    cfg = copy.deepcopy(Config.fromfile(CONFIGS / name).model)
    det = cfg["detector"]
    det["backbone"].update(TINY, init_cfg=None)
    det["neck"].update(in_channels=[32, 64, 128, 256], out_channels=32)
    det["rpn_head"].update(in_channels=32, feat_channels=32)
    det["roi_head"]["bbox_roi_extractor"]["out_channels"] = 32
    det["roi_head"]["bbox_head"].update(in_channels=32, fc_out_channels=64)
    det["test_cfg"]["rpn"].update(nms_pre=200, max_per_img=50)
    det["train_cfg"]["rpn_proposal"].update(nms_pre=200, max_per_img=100)
    return cfg


def meta(frame_id, shape=(96, 128, 3)):
    return dict(img_shape=shape, ori_shape=shape, pad_shape=shape, frame_id=frame_id,
                scale_factor=np.ones(4, dtype=np.float32))


def training_inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        img=torch.randn(1, 3, 96, 128, generator=g), img_metas=[meta(3)],
        gt_bboxes=[torch.tensor([[10.0, 10.0, 60.0, 50.0], [70.0, 20.0, 120.0, 90.0]])],
        gt_labels=[torch.tensor([2, 7])],
        ref_img=torch.randn(1, 4, 3, 96, 128, generator=g),
        ref_img_metas=[[meta(1), meta(5), meta(9), meta(20)]],
    )


@pytest.mark.parametrize("name, temporal", [("tdvit_t_frcnn_fpn_3x.py", True),
                                            ("tdvit_tplus_frcnn_fpn_3x.py", True),
                                            ("frcnn_swint_fpn_3x.py", False)])
def test_each_model_trains_and_every_parameter_learns(name, temporal):
    torch.manual_seed(0)
    model = build_model(tiny_config(name))
    assert isinstance(model, TDViTDetector) and model.temporal == temporal
    model.train()
    losses = model.forward_train(**training_inputs())
    assert {"loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "loss_bbox"} <= set(losses)
    total = sum(sum(v) if isinstance(v, (list, tuple)) else v
                for k, v in losses.items() if "loss" in k)
    assert torch.isfinite(total)
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing  # DDP needs every parameter used on every step


def test_the_model_runs_through_the_trainer_with_accumulation():
    from vfe.engine.trainer import RngStreams, train_step

    torch.manual_seed(0)
    model = build_model(tiny_config()).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    log_vars = train_step(model, [training_inputs(1), training_inputs(2)], optimizer,
                          torch.device("cpu"), rng=RngStreams(2, torch.device("cpu")))
    assert {"loss", "loss_rpn_cls", "loss_cls", "loss_bbox"} <= set(log_vars)
    assert all(np.isfinite(v) for v in log_vars.values())


def test_videos_are_detected_in_order_with_the_memory_reset_per_video():
    torch.manual_seed(0)
    model = build_model(tiny_config()).eval()
    backbone = model.detector.backbone
    video = torch.randn(4, 1, 3, 96, 128)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="first frame"):
            model.simple_test(video[1], [meta(1)])
        lengths = []
        for fid in range(4):
            out = model.simple_test(video[fid], [meta(fid)], rescale=True)
            lengths.append(len(backbone.stages[3].blocks[1].memory))
        assert len(out) == 1 and len(out[0]) == 30
        assert lengths == [1, 2, 3, 4]
        model.simple_test(video[0], [meta(0)])  # the next video
    assert len(backbone.stages[3].blocks[1].memory) == 1
    with pytest.raises(KeyError):
        model.simple_test(video[0], [dict(meta(0), frame_id=-1)])


@pytest.mark.parametrize("name, total", [("frcnn_swint_fpn_3x.py", 44_895_072),
                                         ("tdvit_t_frcnn_fpn_3x.py", 44_895_072),
                                         ("tdvit_tplus_frcnn_fpn_3x.py", 48_448_056)])
def test_the_configs_build(name, total):
    cfg = Config.fromfile(CONFIGS / name)
    model = build_model(cfg.model)
    assert sum(p.numel() for p in model.parameters()) == total
    train = cfg.data.train[0].ref_img_sampler
    assert train.method == "stagewise_uniform" and train.frame_range == [4, 8, 16, 32]
    assert cfg.optimizer.type == "AdamW" and cfg.optimizer.lr == 2.5e-5


# ---- the data ---------------------------------------------------------------------------


@pytest.fixture
def video_dataset(tmp_path):
    images, annotations = [], []
    lengths = {1: 60, 2: 1}  # a long video and a one-frame one
    img_id = 0
    for vid, length in lengths.items():
        for frame_id in range(length):
            img_id += 1
            images.append(dict(id=img_id, file_name=f"v{vid}/{frame_id:06d}.JPEG", width=100,
                               height=80, video_id=vid, frame_id=frame_id,
                               is_vid_train_frame=True))
            annotations.append(dict(id=img_id, image_id=img_id, category_id=1,
                                    bbox=[10, 10, 30, 20], area=600, iscrowd=0,
                                    instance_id=vid))
    ann = dict(categories=[dict(id=i + 1, name=n)
                           for i, n in enumerate(ImagenetVIDDataset.CLASSES)],
               videos=[dict(id=v, name=f"v{v}") for v in lengths],
               images=images, annotations=annotations)
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(ann))
    return ImagenetVIDDataset(str(path), pipeline=[])


SAMPLER = dict(num_ref_imgs=4, frame_range=[4, 8, 16, 32], filter_key_img=True,
               method="stagewise_uniform")


def test_each_stage_gets_a_reference_within_its_dilation(video_dataset):
    key = next(info for info in video_dataset.data_infos if info["frame_id"] == 30)
    offsets = []
    for seed in range(200):
        random.seed(seed)  # the samplers draw from Python's global generator
        key_info, *refs = video_dataset.ref_img_sampling(dict(key), **SAMPLER)
        assert key_info["id"] == key["id"] and len(refs) == 4
        offsets.append([ref["frame_id"] - 30 for ref in refs])
    offsets = np.array(offsets)
    for stage, d in enumerate([4, 8, 16, 32]):
        assert (np.abs(offsets[:, stage]) <= d).all() and (offsets[:, stage] != 0).all()
        assert np.abs(offsets[:, stage]).max() > d // 2  # the whole range is used
    # In stage order, not sorted by frame.
    assert any(not (row == np.sort(row)).all() for row in offsets)


def test_the_window_is_clipped_to_the_video_and_short_videos_reuse_the_key(video_dataset):
    first = next(info for info in video_dataset.data_infos if info["frame_id"] == 0)
    _, *refs = video_dataset.ref_img_sampling(dict(first), **SAMPLER)
    assert all(0 < ref["frame_id"] <= d for ref, d in zip(refs, [4, 8, 16, 32], strict=True))
    lone = next(info for info in video_dataset.data_infos if info["video_id"] == 2)
    _, *refs = video_dataset.ref_img_sampling(dict(lone), **SAMPLER)
    assert [ref["id"] for ref in refs] == [lone["id"]] * 4
    with pytest.raises(ValueError, match="one reference per range"):
        video_dataset.ref_img_sampling(dict(lone), **dict(SAMPLER, num_ref_imgs=2))


def coordinate_frame(index, h=500, w=900):
    ys, xs = np.mgrid[:h, :w]
    img = np.stack([ys % 256, xs % 256, np.full_like(ys, index)], axis=-1).astype(np.uint8)
    return {"img": img, "img_fields": ["img"], "img_shape": img.shape,
            "bbox_fields": ["gt_bboxes"], "gt_labels": np.array([3]),
            "gt_bboxes": np.array([[100.0, 100.0, 300.0, 250.0]], dtype=np.float32)}


def test_a_shared_crop_cuts_every_frame_of_the_clip_alike():
    crop = SeqRandomCrop(crop_size=(384, 600), crop_type="absolute_range",
                         allow_negative_crop=True, share_params=True)
    np.random.seed(0)
    for _ in range(20):
        out = crop([coordinate_frame(i) for i in range(5)])
        key = out[0]
        for i, frame in enumerate(out):
            assert frame["img"].shape == key["img"].shape == frame["img_shape"]
            assert np.array_equal(frame["img"][..., :2], key["img"][..., :2])  # same window
            assert (frame["img"][..., 2] == i).all()
            assert np.array_equal(frame["gt_bboxes"], key["gt_bboxes"])


def test_an_independent_crop_still_cuts_each_frame_on_its_own():
    crop = SeqRandomCrop(crop_size=(384, 600), crop_type="absolute_range",
                         allow_negative_crop=True)
    np.random.seed(0)
    out = crop([coordinate_frame(i) for i in range(5)])
    windows = {(f["img"].shape, int(f["img"][0, 0, 0]), int(f["img"][0, 0, 1])) for f in out}
    assert len(windows) > 1


# ---- SELSA on TDViT (Table 3) -----------------------------------------------------------


def tiny_selsa(name="selsa_tdvit_t_fpn_3x.py"):
    cfg = tiny_config(name)
    head = cfg["detector"]["roi_head"]["bbox_head"]
    head.update(topk=10)
    head["aggregator"].update(in_channels=64, num_attention_blocks=4)
    torch.manual_seed(0)
    return build_model(cfg)


def selsa_inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    inputs = training_inputs(seed)
    inputs["ref_img"] = torch.randn(1, 6, 3, 96, 128, generator=g)
    inputs["ref_img_metas"] = [[meta(i) for i in (1, 5, 9, 20, 2, 4)]]
    return inputs


def test_forward_spatial_is_the_first_frame_and_leaves_the_memory_alone():
    tdvit = tiny_tdvit().eval()
    frames = torch.randn(3, 3, 72, 104)
    with torch.no_grad():
        tdvit(frames[:1])  # one frame into the memories
        spatial = tdvit.forward_spatial(frames[1:])
        lengths = [len(b.memory) for s in tdvit.stages for b in s.blocks if isinstance(b, TDTB)]
        first = []
        for i in (1, 2):  # each frame as a video's first: every TDTB attends to it
            tdvit.reset_memory()
            first.append(tdvit(frames[i:i + 1]))
    assert lengths == [1] * 6
    for level, (a, b) in enumerate(zip(*first, strict=True)):
        torch.testing.assert_close(spatial[level], torch.cat((a, b)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["selsa_tdvit_t_fpn_3x.py", "selsa_swint_fpn_3x.py"])
def test_selsa_trains_and_every_parameter_learns(name):
    model = tiny_selsa(name).train()
    assert model.selsa and model.backbone_refs == 4
    losses = model.forward_train(**selsa_inputs())
    total = sum(sum(v) if isinstance(v, (list, tuple)) else v
                for k, v in losses.items() if "loss" in k)
    assert torch.isfinite(total)
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_the_swin_baseline_skips_tdvits_references():
    model = tiny_selsa("selsa_swint_fpn_3x.py").train()
    inputs = selsa_inputs()
    changed = dict(inputs, ref_img=inputs["ref_img"].clone())
    changed["ref_img"][:, :4] = torch.randn(1, 4, 3, 96, 128)
    torch.manual_seed(7)
    a = model.forward_train(**inputs)
    torch.manual_seed(7)
    b = model.forward_train(**changed)
    assert all(torch.equal(a[k], b[k]) for k in ("loss_cls", "loss_bbox"))
    changed["ref_img"][:, 4:] = torch.randn(1, 2, 3, 96, 128)
    torch.manual_seed(7)
    c = model.forward_train(**changed)
    assert not torch.equal(a["loss_cls"], c["loss_cls"])


def test_selsa_tests_with_the_videos_references_and_its_own_rois():
    model = tiny_selsa().eval()
    video, refs = torch.randn(4, 1, 3, 96, 128), torch.randn(1, 3, 3, 96, 128)
    ref_metas = [[[meta(0), meta(4), meta(8)]]]
    with torch.no_grad():
        with pytest.raises(ValueError, match="references"):
            model.simple_test(video[0], [meta(0)])
        out = model.simple_test(video[0], [meta(0)], ref_img=[refs], ref_img_metas=ref_metas,
                                rescale=True)
        ref_feats, ref_rois = model._refs
        assert len(ref_rois) == 3 and all(len(r) <= 10 for r in ref_rois)
        assert ref_feats[0].shape[0] == 3
        for fid in (1, 2, 3):
            out = model.simple_test(video[fid], [meta(fid)], rescale=True)
        assert len(out) == 1 and len(out[0]) == 30
        assert len(model.detector.backbone.stages[3].blocks[1].memory) == 4
        model.simple_test(video[0], [meta(0)], ref_img=[refs[:, :2]],
                          ref_img_metas=[[ref_metas[0][0][:2]]])
    assert len(model._refs[1]) == 2  # a new video, its own references


def test_the_selsa_configs_keep_tdvit_ts_pipelines():
    tdvit = Config.fromfile(CONFIGS / "tdvit_t_frcnn_fpn_3x.py").data
    for name in ("selsa_tdvit_t_fpn_3x.py", "selsa_swint_fpn_3x.py"):
        selsa = Config.fromfile(CONFIGS / name).data
        for a, b in zip(selsa.train, tdvit.train, strict=True):
            assert a.pipeline == b.pipeline and a.ann_file == b.ann_file
            assert a.ref_img_sampler.frame_range[:4] == b.ref_img_sampler.frame_range
        assert selsa.test.pipeline == tdvit.test.pipeline
        assert selsa.test.ref_img_sampler.method == "test_with_adaptive_stride"


def test_offline_testing_sees_every_frame_on_its_own():
    torch.manual_seed(0)
    online = build_model(tiny_config()).eval()
    offline = build_model(dict(tiny_config(), online=False)).eval()
    offline.load_state_dict(online.state_dict())
    video = torch.randn(3, 1, 3, 96, 128)
    with torch.no_grad():
        outs = [offline.simple_test(video[i], [meta(i)]) for i in range(3)]
        assert len(offline.detector.backbone.stages[3].blocks[1].memory) == 0
        for i in range(3):  # each as the first frame of a video, online
            first = online.simple_test(video[i], [meta(0)])
            for a, b in zip(outs[i][0], first[0], strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


def test_offset_windows_take_past_references_and_clip_at_the_start(video_dataset):
    sampler = dict(SAMPLER, frame_range=[[-7, -4], [-15, -8], [-31, -16], [-63, -32]])
    by_frame = {info["frame_id"]: info for info in video_dataset.data_infos
                if info["video_id"] == 1}
    random.seed(0)
    for _ in range(50):
        _, *refs = video_dataset.ref_img_sampling(dict(by_frame[59]), **sampler)
        offsets = [ref["frame_id"] - 59 for ref in refs]
        for off, (lo, hi) in zip(offsets, sampler["frame_range"], strict=True):
            assert lo <= off <= hi or (59 + lo < 0 and off == -59)
    # Near the start a window clips to the video, or falls back to frame 0 --
    # what the first frames attend to at test time -- and frame 0 to itself.
    _, *refs = video_dataset.ref_img_sampling(dict(by_frame[5]), **sampler)
    assert [ref["frame_id"] for ref in refs][1:] == [0, 0, 0]
    assert 0 <= refs[0]["frame_id"] <= 1
    _, *refs = video_dataset.ref_img_sampling(dict(by_frame[0]), **sampler)
    assert [ref["id"] for ref in refs] == [by_frame[0]["id"]] * 4
    with pytest.raises(ValueError, match="windows"):
        video_dataset.ref_img_sampling(dict(by_frame[0]), **dict(sampler, frame_range=[[-4, -8]] * 4))


# ---- joint attention: the frame's window and the reference's together ---------------------


@pytest.mark.parametrize("shift", [False, True])
def test_joint_attention_with_the_frame_as_reference_is_swin_whatever_the_bias(shift):
    tdtb = block(shift=shift, attention="joint", temporal_bias=True)
    torch.nn.init.normal_(tdtb.attn.w_msa.temporal_bias, std=2.0)
    swin = block(SwinBlock, shift=shift)
    swin.load_state_dict(tdtb.state_dict(), strict=False)
    x = torch.randn(2, 10 * 11, 16)
    with torch.no_grad():
        torch.testing.assert_close(tdtb(x, (10, 11), ref=x), swin(x, (10, 11)),
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shift", [False, True])
def test_the_temporal_bias_moves_joint_attention_between_swin_and_cross(shift):
    joint = block(shift=shift, attention="joint", temporal_bias=True)
    cross = block(shift=shift)
    cross.load_state_dict(joint.state_dict(), strict=False)
    swin = block(SwinBlock, shift=shift)
    swin.load_state_dict(joint.state_dict(), strict=False)
    x, ref = torch.randn(1, 10 * 11, 16), torch.randn(1, 10 * 11, 16)
    with torch.no_grad():
        joint.attn.w_msa.temporal_bias.fill_(-1e4)  # the reference ignored
        torch.testing.assert_close(joint(x, (10, 11), ref=ref), swin(x, (10, 11)),
                                   rtol=1e-5, atol=1e-5)
        joint.attn.w_msa.temporal_bias.fill_(50.0)  # the frame's own keys ignored (e^-50)
        torch.testing.assert_close(joint(x, (10, 11), ref=ref), cross(x, (10, 11), ref=ref),
                                   rtol=1e-5, atol=1e-5)
        joint.attn.w_msa.temporal_bias.zero_()
        out = joint(x, (10, 11), ref=ref)
    assert not torch.allclose(out, swin(x, (10, 11))) and not torch.allclose(out, cross(x, (10, 11), ref=ref))


def test_joint_online_frames_match_a_recomputation():
    tdtb = block(shift=True, temporal_dilation=3, attention="joint", temporal_bias=True)
    torch.nn.init.normal_(tdtb.attn.w_msa.temporal_bias)
    video = [torch.randn(1, 10 * 11, 16) for _ in range(8)]
    shadow = MemoryQueue(3)
    with torch.no_grad():
        for x in video:
            ref = shadow.sample(x, (10, 11))
            shadow.update(x)
            expected = tdtb(x, (10, 11)) if ref is x else tdtb(x, (10, 11), ref=ref)
            torch.testing.assert_close(tdtb.forward_online(x, (10, 11)), expected,
                                       rtol=1e-5, atol=1e-5)


def test_joint_tdvit_adds_only_the_temporal_biases_and_all_of_it_learns():
    tdvit = tiny_tdvit(attention="joint", temporal_bias=True).train()
    swin_keys = set(SwinTransformer(depths=[2, 2, 6, 2], **TINY).state_dict())
    extra = sorted(set(tdvit.state_dict()) - swin_keys)
    assert extra and all(k.endswith("attn.w_msa.temporal_bias") for k in extra)
    assert len(extra) == 6  # one per TDTB
    outs = tdvit(torch.randn(1, 3, 72, 104), torch.randn(1, 4, 3, 72, 104))
    sum(o.square().mean() for o in outs).backward()
    assert all(p.grad is not None for p in tdvit.parameters())
    with pytest.raises(ValueError, match="joint"):
        TDViT(**TINY, temporal_bias=True)


@pytest.mark.parametrize("attention", ["cross", "joint"])
def test_zero_initialised_extra_tdtbs_start_as_the_identity(attention):
    torch.manual_seed(0)
    base = TDViT(**TINY, attention=attention)
    base.init_weights()
    plus = TDViT(**TINY, attention=attention, extra_tdtbs=(0, 0, 2, 0), extra_init="zero")
    plus.init_weights()
    plus.load_state_dict(base.state_dict(), strict=False)  # the blocks they share
    for block in plus.stages[2].blocks[6:]:
        assert not block.attn.w_msa.proj.weight.any() and not block.ffn.layers[1].weight.any()
    base.eval(), plus.eval()
    img, refs = torch.randn(1, 3, 72, 104), torch.randn(1, 4, 3, 72, 104)
    with torch.no_grad():
        for a, b in zip(base(img, refs), plus(img, refs), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    # They still learn: the zeroed projections get gradients at once.
    plus.train()
    sum(o.square().mean() for o in plus(img, refs)).backward()
    assert plus.stages[2].blocks[6].attn.w_msa.proj.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="extra_init"):
        TDViT(**TINY, extra_init="copy")


@pytest.mark.parametrize("backbone", ["swin", "cross", "joint"])
def test_fused_attention_matches_the_unfused(backbone):
    torch.manual_seed(0)
    if backbone == "swin":
        plain = SwinTransformer(depths=[2, 2, 6, 2], **TINY).eval()
        fused = SwinTransformer(depths=[2, 2, 6, 2], fused_attention=True, **TINY).eval()
    else:
        plain = tiny_tdvit(attention=backbone, temporal_bias=backbone == "joint").eval()
        fused = TDViT(**TINY, attention=backbone, temporal_bias=backbone == "joint",
                      fused_attention=True).eval()
        for p in plain.parameters():
            if p.dim() == 1 and p.shape[0] < 30:  # the temporal biases: make them matter
                torch.nn.init.normal_(p)
    fused.load_state_dict(plain.state_dict())
    assert all(m.fused for m in fused.modules() if hasattr(m, "fused"))
    video = torch.randn(3, 1, 3, 72, 104)  # padded and shifted windows; online frames
    with torch.no_grad():
        for frame in video:
            for a, b in zip(plain(frame), fused(frame), strict=True):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

"""MAMBA's pixel level: key pixels from boxes, the memory writes, and the
detector's three variants (full, pixel-only, baseline) training and running
stateful inference on tiny models built from the real configs."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from vfe.config import Config
from vfe.models.builder import build_model
from vfe.models.roi_heads.mamba import MambaRoIHead
from vfe.models.vid.eovod import boxes_to_level_masks
from vfe.models.vid.mamba import MAMBA, MambaPixelLevel

REPO_ROOT = Path(__file__).resolve().parents[1]


def tiny_config(name: str = "mamba_full_r101_dc5_3x.py") -> dict:
    """A config from configs/vid/mamba shrunk to ResNet-18 and 32 channels."""
    cfg = copy.deepcopy(Config.fromfile(REPO_ROOT / "configs/vid/mamba" / name).model)
    det = cfg["detector"]
    det["backbone"].update(depth=18, init_cfg=None)
    det["neck"].update(in_channels=[512], out_channels=32)
    det["rpn_head"].update(in_channels=32, feat_channels=32)
    det["roi_head"]["bbox_roi_extractor"]["out_channels"] = 32
    head = det["roi_head"]["bbox_head"]
    head.update(in_channels=32, fc_out_channels=64)
    if "aggregator" in head:
        head["topk"] = 10
        head["aggregator"].update(in_channels=64, num_attention_blocks=4)
    if cfg.get("pixel") is not None:
        channels = 512 if cfg["pixel"]["position"] == "backbone" else 32
        cfg["pixel"].update(in_channels=channels, num_attention_blocks=4)
    det["test_cfg"]["rpn"].update(nms_pre=200, max_per_img=50)
    det["train_cfg"]["rpn_proposal"].update(nms_pre=200, max_per_img=100)
    return cfg


def tiny_mamba(name: str = "mamba_full_r101_dc5_3x.py", **pixel) -> MAMBA:
    cfg = tiny_config(name)
    if pixel:
        cfg["pixel"].update(pixel)
    torch.manual_seed(0)
    return build_model(cfg)


def meta(frame_id, shape=(96, 128, 3), scale=1.0):
    return dict(img_shape=shape, ori_shape=shape, pad_shape=shape, frame_id=frame_id,
                scale_factor=np.full(4, scale, dtype=np.float32))


def level(**kwargs) -> MambaPixelLevel:
    kwargs.setdefault("in_channels", 8)
    kwargs.setdefault("num_attention_blocks", 2)
    return MambaPixelLevel(**kwargs)


# ---- the pixel level on its own ---------------------------------------------------------

def test_box_cells_are_the_cells_whose_centres_lie_inside():
    pixel = level(stride=16)
    cells = pixel.box_cells(torch.tensor([[8.0, 8.0, 40.0, 24.0], [30.0, 30.0, 31.0, 31.0]]), 4, 8)
    # Centres x in {8, 24, 40}, y in {8, 24}; the tiny box keeps the cell holding it.
    assert cells[0].tolist() == [0, 1, 2, 8, 9, 10]
    assert cells[1].tolist() == [9]
    # The same rule as EOVOD's masks, box by box.
    torch.manual_seed(0)
    xy = torch.rand(20, 2) * torch.tensor([128.0, 64.0])
    boxes = torch.cat([xy, xy + torch.rand(20, 2) * 60 + 1], dim=1)
    for box, got in zip(boxes, pixel.box_cells(boxes, 4, 8), strict=True):
        mask = boxes_to_level_masks(box[None], [(4, 8)], [16])[0]
        assert got.tolist() == mask.flatten().nonzero().squeeze(1).tolist()


def test_pixels_take_k_per_box_cap_the_frame_and_fall_back_to_the_strongest():
    pixel = level(stride=16, pixels_per_box=2, pixels_per_frame=3, fallback_pixels=2)
    feat = torch.arange(3 * 4 * 8, dtype=torch.float32).view(3, 4, 8)
    boxes = torch.tensor([[8.0, 8.0, 40.0, 24.0], [70.0, 40.0, 120.0, 60.0]])
    torch.manual_seed(0)
    rows = pixel.pixels(feat, boxes)
    assert rows.shape == (3, 3)
    cells = [int(r[0]) for r in rows]  # channel 0 holds the flat index
    first_box = set(pixel.box_cells(boxes[:1], 4, 8)[0].tolist())
    second_box = set(pixel.box_cells(boxes[1:], 4, 8)[0].tolist())
    assert set(cells[:2]) <= first_box and len(set(cells[:2])) == 2
    assert cells[2] in second_box
    assert torch.equal(rows, feat.flatten(1)[:, cells].t())
    # No box: the highest-norm cells, i.e. the last two here.
    assert sorted(int(r[0]) for r in pixel.pixels(feat, torch.zeros(0, 4))) == [30, 31]


def test_writes_keep_confident_detections_highest_first():
    pixel = level(stride=16, score_thr=0.3, pixels_per_box=1, fallback_pixels=1)
    feat = torch.arange(3 * 4 * 8, dtype=torch.float32).view(3, 4, 8)
    dets = torch.tensor([[0.0, 0.0, 15.0, 15.0, 0.2],     # below the threshold
                         [96.0, 48.0, 127.0, 63.0, 0.5],  # cell (3, 6) and (3, 7)
                         [16.0, 16.0, 31.0, 31.0, 0.9]])  # cell (1, 1)
    pixel.write(feat, dets)
    assert [int(r[0]) for r in pixel.memory.feat] == [9, 30] or \
        [int(r[0]) for r in pixel.memory.feat] == [9, 31]
    pixel.write(feat, dets[:1])  # nothing confident: the strongest pixel
    assert int(pixel.memory.feat[-1, 0]) == 31
    assert not pixel.memory.feat.requires_grad
    pixel.reset()
    assert pixel.memory.feat is None


def test_training_keys_from_ground_truth_or_at_random():
    feats = torch.randn(2, 8, 4, 8)
    ref_boxes = torch.tensor([[0.0, 8.0, 8.0, 40.0, 24.0]])  # reference 0 only
    gt = level(stride=16, train_keys="gt", pixels_per_box=100, fallback_pixels=3)
    keys = gt.training_keys(feats, ref_boxes)
    assert keys.shape == (6 + 3, 8)  # six cells in the box; three fallback pixels of ref 1
    rand = level(stride=16, train_keys="random", random_keys=5,
                 memory_cfg=dict(key_length=7))
    assert rand.training_keys(feats, ref_boxes).shape == (7, 8)  # 2 x 5, capped at 7
    with pytest.raises(ValueError, match="train_keys"):
        level(train_keys="boxes")
    with pytest.raises(ValueError, match="position"):
        level(position="fpn")


def test_every_pixel_attends_and_the_result_is_residual():
    pixel = level()
    feat = torch.randn(1, 8, 3, 5)
    keys = torch.randn(4, 8)
    out = pixel(feat, keys)
    assert out.shape == feat.shape and not torch.equal(out, feat)
    query = feat.flatten(2)[0].t()
    expected = query + pixel.aggregator.forward_with_ref_x(query, keys)
    assert torch.allclose(out.flatten(2)[0].t(), expected)
    torch.nn.init.zeros_(pixel.aggregator.fc.weight)
    torch.nn.init.zeros_(pixel.aggregator.fc.bias)
    assert torch.equal(pixel(feat, keys), feat)


# ---- the detector ---------------------------------------------------------------------

VARIANTS = {
    "full": "mamba_full_r101_dc5_3x.py",
    "pix": "mamba_pix_r101_dc5_3x.py",
    "pix_neck": "mamba_pix_neck_r101_dc5_3x.py",
    "baseline": "frcnn_r101_dc5_3x.py",
}


def training_inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        img=torch.randn(1, 3, 96, 128, generator=g), img_metas=[meta(3)],
        gt_bboxes=[torch.tensor([[10.0, 10.0, 60.0, 50.0], [70.0, 20.0, 120.0, 90.0]])],
        gt_labels=[torch.tensor([2, 7])], gt_instance_ids=[torch.tensor([0, 1])],
        ref_img=torch.randn(1, 2, 3, 96, 128, generator=g), ref_img_metas=[[meta(1), meta(5)]],
        ref_gt_bboxes=[torch.tensor([[0.0, 12.0, 8.0, 58.0, 52.0], [1.0, 68.0, 22.0, 118.0, 88.0]])],
        ref_gt_labels=[torch.tensor([[0, 2], [1, 7]])],
    )


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_each_variant_trains_and_every_parameter_learns(variant):
    model = tiny_mamba(VARIANTS[variant])
    assert (model.pixel is not None) == (variant != "baseline")
    assert model.instance_level == (variant == "full")
    model.train()
    losses = model.forward_train(**training_inputs())
    assert {"loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "loss_bbox"} <= set(losses)
    total = sum(sum(v) if isinstance(v, (list, tuple)) else v
                for k, v in losses.items() if "loss" in k)
    assert torch.isfinite(total)
    total.backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None]
    assert not missing, missing  # DDP needs every parameter used on every step
    # Reference frames without boxes still give the pixel level keys.
    if model.pixel is not None:
        model.zero_grad()
        inputs = dict(training_inputs(2), ref_gt_bboxes=[torch.zeros(0, 5)])
        sum(sum(v) if isinstance(v, (list, tuple)) else v
            for k, v in model.forward_train(**inputs).items() if "loss" in k).backward()
        assert model.pixel.aggregator.fc.weight.grad is not None


def test_the_full_model_runs_through_the_trainer_with_accumulation():
    from vfe.engine.trainer import RngStreams, train_step

    model = tiny_mamba()
    model.train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    torch.manual_seed(0)
    log_vars = train_step(model, [training_inputs(1), training_inputs(2)], optimizer,
                          torch.device("cpu"), rng=RngStreams(2, torch.device("cpu")))
    assert {"loss", "loss_rpn_cls", "loss_cls", "loss_bbox"} <= set(log_vars)
    assert all(np.isfinite(v) for v in log_vars.values())


def video_frames(n, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, 3, 96, 128, generator=g), torch.randn(1, 3, 3, 96, 128, generator=g)


@pytest.mark.parametrize("variant", ["full", "pix", "pix_neck"])
def test_stateful_inference_fills_and_resets_the_pixel_memory(variant):
    # score_thr 0 validates every detection, so random weights still write boxes.
    model = tiny_mamba(VARIANTS[variant], score_thr=0.0, pixels_per_box=5,
                       pixels_per_frame=20).eval()
    frames, refs = video_frames(4)
    ref_metas = [[[meta(0), meta(4), meta(8)]]]
    with torch.no_grad():
        out = model.simple_test(frames[0], [meta(0)], ref_img=[refs], ref_img_metas=ref_metas,
                                rescale=True)
        assert len(out) == 1 and len(out[0]) == 30
        assert all(arr.shape[1] == 5 for arr in out[0])
        sizes = [len(model.pixel.memory)]
        for fid in (1, 2, 3):
            model.simple_test(frames[fid], [meta(fid)], rescale=True)
            sizes.append(len(model.pixel.memory))
    # Frame 0: three references and the key frame, then the key frame again,
    # each at most 20 pixels; every later frame adds its own.
    assert 5 <= sizes[0] <= 5 * 20
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]
    if model.instance_level:
        assert all(agg.memory_bank.feat is not None for agg in
                   model.detector.roi_head.bbox_head.aggregator)
    # A new video starts from its own references only.
    with torch.no_grad():
        model.simple_test(frames[1], [meta(0)], ref_img=[refs], ref_img_metas=ref_metas)
    assert len(model.pixel.memory) <= 5 * 20


def test_writes_use_input_coordinates_when_the_output_is_rescaled():
    model = tiny_mamba(score_thr=0.0).eval()
    frames, refs = video_frames(2)
    written = []
    original = model.pixel.write

    def spy(feat, det_bboxes):
        written.append(det_bboxes.clone())
        original(feat, det_bboxes)

    model.pixel.write = spy
    with torch.no_grad():
        model.simple_test(frames[0], [meta(0)], ref_img=[refs],
                          ref_img_metas=[[[meta(0), meta(4), meta(8)]]])
        written.clear()
        out = model.simple_test(frames[1], [meta(1, scale=0.5)], rescale=True)
    returned = torch.from_numpy(np.concatenate(out[0]))
    assert len(written) == 1 and len(returned) == len(written[0])
    order = returned[:, 4].argsort()
    assert torch.allclose(returned[order, :4] * 0.5, written[0][written[0][:, 4].argsort(), :4])


def test_pixel_inference_errors():
    model = tiny_mamba().eval()
    frames, refs = video_frames(1)
    with torch.no_grad():
        with pytest.raises(ValueError, match="reference frames"):
            model.simple_test(frames[0], [meta(0)])
        with pytest.raises(KeyError):
            model.simple_test(frames[0], [dict(meta(0), frame_id=-1)])
        with pytest.raises(NotImplementedError):
            model.simple_test(frames[0], [dict(meta(0), frame_stride=10)])


def test_the_baseline_detects_frame_by_frame():
    model = tiny_mamba(VARIANTS["baseline"]).eval()
    frames, _ = video_frames(2)
    with torch.no_grad():
        first = model.simple_test(frames[0], [meta(0)], rescale=True)
        again = model.simple_test(frames[0], [meta(1)], rescale=True)
    assert len(first[0]) == 30
    assert all(np.array_equal(a, b) for a, b in zip(first[0], again[0], strict=True))


def test_released_checkpoint_keys_are_unchanged():
    """No pixel level: the released MAMBA's parameters exactly, so its
    checkpoint still loads with nothing missing or unexpected."""
    released = build_model(tiny_config("mamba_full_r101_dc5_3x.py") | {"pixel": None})
    assert isinstance(released.detector.roi_head, MambaRoIHead) and released.pixel is None
    assert not any(k.startswith("pixel.") for k in released.state_dict())
    full = tiny_mamba()
    extra = set(full.state_dict()) - set(released.state_dict())
    assert extra and all(k.startswith("pixel.aggregator.") for k in extra)


@pytest.mark.parametrize("config, roi_head, pixel_position", [
    ("mamba_full_r101_dc5_3x.py", "MambaRoIHead", "backbone"),
    ("mamba_full_r101_dc5_6x.py", "MambaRoIHead", "backbone"),
    ("mamba_pix_r101_dc5_3x.py", "StandardRoIHead", "backbone"),
    ("mamba_pix_neck_r101_dc5_3x.py", "StandardRoIHead", "neck"),
    ("frcnn_r101_dc5_3x.py", "StandardRoIHead", None),
])
def test_mamba_variant_configs_build(config, roi_head, pixel_position):
    model = build_model(Config.fromfile(REPO_ROOT / "configs/vid/mamba" / config).model)
    assert type(model.detector.roi_head).__name__ == roi_head
    assert (model.pixel.position if model.pixel is not None else None) == pixel_position
    released = 89_620_755  # tests/test_models.py
    instance = 2 * 4 * (1024 * 1024 + 1024)  # two MambaAggregators on the 1024-wide FCs
    channels = {"backbone": 2048, "neck": 512, None: 0}[pixel_position]
    pixel = 4 * (channels * channels + channels)
    expected = released - (0 if roi_head == "MambaRoIHead" else instance) + pixel
    assert sum(p.numel() for p in model.parameters()) == expected

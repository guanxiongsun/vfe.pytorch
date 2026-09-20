"""Model construction from the real configs, and STPN's prompted Swin."""

from pathlib import Path

import pytest
import torch

from vfe.config import Config
from vfe.models.backbones import STPNSwinTransformer, SwinTransformer
from vfe.models.builder import build_model
from vfe.models.vid.stpn import AttentionPredictor

REPO_ROOT = Path(__file__).resolve().parents[1]

SWIN_T = dict(embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
              mlp_ratio=4, qkv_bias=True, drop_path_rate=0.0, patch_norm=True)
PROMPTS = dict(num_tokens=5, location="prepend", deep=False, dropout=0.0, initiation="random")


@pytest.mark.parametrize("config, model_type, total, trainable", [
    ("configs/vid/mamba/mamba_r101_dc5_6x.py", "MAMBA", 89_620_755, 89_395_411),
    ("configs/vid/stpn/stpn_swint_adam_9x.py", "STPN", 45_006_624, 45_006_624),
    # Swin-S: the Swin-T config with stage 3 at 18 blocks instead of 6.
    ("configs/vid/stpn/stpn_swins_adam_9x.py", "STPN", 66_324_528, 66_324_528),
])
def test_models_build_from_their_configs(config, model_type, total, trainable):
    model = build_model(Config.fromfile(REPO_ROOT / config).model)
    assert type(model).__name__ == model_type
    assert sum(p.numel() for p in model.parameters()) == total
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == trainable


def test_prompted_swin_without_prompts_is_plain_swin():
    torch.manual_seed(0)
    prompted = STPNSwinTransformer(prompt_cfg=PROMPTS, **SWIN_T).eval()
    plain = SwinTransformer(**SWIN_T).eval()
    plain.load_state_dict(prompted.state_dict())  # prompts add no parameters
    x = torch.randn(1, 3, 64, 96)  # 16x24 tokens: padded, shifted windows
    with torch.no_grad():
        for a, b in zip(prompted(x), plain(x), strict=True):
            assert torch.equal(a, b)


def test_prompts_travel_through_every_stage_and_leave_the_outputs():
    torch.manual_seed(0)
    backbone = STPNSwinTransformer(prompt_cfg=PROMPTS, **SWIN_T).eval()
    x = torch.randn(1, 3, 64, 96)
    prompts = torch.randn(5, 96)
    with torch.no_grad():
        plain_outs = backbone(x)
        prompted_outs = backbone(x, prompts)
    for plain, prompted in zip(plain_outs, prompted_outs, strict=True):
        assert plain.shape == prompted.shape  # prompt tokens removed from the feature maps
        assert not torch.equal(plain, prompted)  # but they changed the features


def test_attention_predictor_output():
    torch.manual_seed(0)
    predictor = AttentionPredictor(768, num_prompts=5, prompt_dims=96)
    ref_features = torch.randn(3, 768, 4, 6)  # 3 reference frames
    assert predictor(ref_features).shape == (5, 96)

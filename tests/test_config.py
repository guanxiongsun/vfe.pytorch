"""Config loading and command-line overrides (``--cfg-options``)."""

from pathlib import Path

import pytest

from vfe.config import Config, parse_cfg_options

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parse_cfg_options_types():
    options = parse_cfg_options([
        "a=(500,300)", "b=[1,[2,3]]", "c=x,y", "d=/data/ann.json", "e=true", "f=1e-3",
        "g=[]", "h=(1)", "i=None", "j=\"q,r\"", "k=((1,2),(3,4))", "l=12",
    ])
    assert options == {
        "a": (500, 300), "b": [1, [2, 3]], "c": ["x", "y"], "d": "/data/ann.json", "e": True,
        "f": 0.001, "g": [], "h": (1,), "i": None, "j": "q,r", "k": ((1, 2), (3, 4)), "l": 12,
    }


def test_parse_cfg_options_malformed_values_stay_strings():
    # Unbalanced brackets once sent the parser into infinite recursion.
    assert parse_cfg_options(["x=abc) d=(5"]) == {"x": "abc) d=(5"}
    with pytest.raises(ValueError):
        parse_cfg_options(["no_equals_sign"])


def test_merge_from_dict_list_indices_and_dicts():
    cfg = Config.fromfile(REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_6x.py")
    det_ann_file = cfg.data.train[1].ann_file
    cfg.merge_from_dict(parse_cfg_options([
        "data.train.0.ann_file=/tmp/vid.json",
        "data.train.0.pipeline.2.img_scale=(500,300)",
        "log_config.interval=1",
        "brand.new.key=7",
    ]))
    cfg.merge_from_dict({"optimizer": {"lr": 0.5}})  # a dict merges into the existing dict

    assert isinstance(cfg.data.train, list)
    assert cfg.data.train[0].ann_file == "/tmp/vid.json"
    assert cfg.data.train[1].ann_file == det_ann_file  # the other list item is untouched
    assert cfg.data.train[0].pipeline[2] == {"type": "SeqResize", "img_scale": (500, 300),
                                             "keep_ratio": True}
    assert cfg.log_config.interval == 1
    assert cfg.brand.new.key == 7
    assert cfg.optimizer == {"type": "SGD", "lr": 0.5, "momentum": 0.9, "weight_decay": 0.0001}


def test_stpn_config_inherits_and_deletes():
    cfg = Config.fromfile(REPO_ROOT / "configs/vid/stpn/stpn_swint_adam_9x.py")
    # `_delete_=True` replaced the base schedule's SGD optimiser outright.
    assert cfg.optimizer["type"] == "AdamW"
    assert "momentum" not in cfg.optimizer
    assert cfg.optimizer_config == {"grad_clip": None}
    assert [t["type"] for t in cfg.data.train[0].pipeline][3] == "AutoAugment"

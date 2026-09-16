"""Swin Transformer backbone (https://arxiv.org/abs/2103.14030).

Ported from mmdet 2.19.1's implementation, which is *not* identical to the
original Microsoft one -- it uses ``nn.Unfold`` for patch merging and splits
the MLP into mmcv's ``FFN``. ``swin_convert`` below translates the upstream
checkpoint layout into this one; the ``convert_weights=True`` flag in the STPN
configs is what triggers it.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn

from ...layers import build_norm_layer, constant_init, trunc_normal_init
from ...layers.drop import build_dropout
from ...layers.transformer import FFN, PatchEmbed, PatchMerging, to_2tuple
from ..builder import BACKBONES

__all__ = [
    "SwinTransformer",
    "SwinBlock",
    "SwinBlockSequence",
    "ShiftWindowMSA",
    "WindowMSA",
    "swin_convert",
]


class WindowMSA(nn.Module):
    """Window-based multi-head self-attention with a relative position bias."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: tuple[int, int],
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # mmdet's closed-form construction of the relative-position index --
        # ~2x faster than the meshgrid version and produces the same table.
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: (num_windows*B, N, C); ``mask``: (num_windows, N, N) in (-inf, 0]."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    @staticmethod
    def double_step_seq(step1: int, len1: int, step2: int, len2: int) -> torch.Tensor:
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):
    """Window attention with the cyclic shift that lets windows exchange info.

    When ``shift_size > 0`` the feature map is rolled and an attention mask
    stops tokens that wrapped around from attending to each other.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        dropout_layer: dict | None = None,
    ):
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError(f"shift_size {shift_size} out of range for window {window_size}")
        self.window_size = window_size
        self.shift_size = shift_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.drop = build_dropout(dropout_layer or {"type": "DropPath", "drop_prob": 0.0})

    def forward(self, query: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
        B, L, C = query.shape
        H, W = hw_shape
        if L != H * W:
            raise ValueError(f"input has {L} tokens but hw_shape {hw_shape} implies {H * W}")
        query = query.view(B, H, W, C)

        # Pad up to a whole number of windows.
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if self.shift_size > 0:
            shifted_query = torch.roll(
                query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            # Label each region by where it came from, so tokens from different
            # regions that now share a window can be masked apart.
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in slices:
                for w in slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            shifted_query = query
            attn_mask = None

        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size**2, C)

        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        return self.drop(x.view(B, H * W, C))

    def window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, window_size, window_size, C)


class SwinBlock(nn.Module):
    """LN -> (shifted) window attention -> residual -> LN -> FFN -> residual."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        with_cp: bool = False,
    ):
        super().__init__()
        act_cfg = act_cfg or {"type": "GELU"}
        norm_cfg = norm_cfg or {"type": "LN"}
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer={"type": "DropPath", "drop_prob": drop_path_rate},
        )
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer={"type": "DropPath", "drop_prob": drop_path_rate},
            act_cfg=act_cfg,
            add_identity=True,
        )

    def forward(self, x: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            return self.ffn(x, identity=identity)

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x, use_reentrant=False)
        return _inner_forward(x)


class SwinBlockSequence(nn.Module):
    """One Swin stage: ``depth`` blocks with alternating shift, then downsample."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float | list[float] = 0.0,
        downsample: nn.Module | None = None,
        act_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        with_cp: bool = False,
    ):
        super().__init__()
        if isinstance(drop_path_rate, list):
            if len(drop_path_rate) != depth:
                raise ValueError(f"expected {depth} drop path rates, got {len(drop_path_rate)}")
            drop_path_rates = drop_path_rate
        else:
            drop_path_rates = [drop_path_rate] * depth

        self.blocks = nn.ModuleList(
            SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=bool(i % 2),
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
            )
            for i in range(depth)
        )
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        return x, hw_shape, x, hw_shape


@BACKBONES.register_module()
class SwinTransformer(nn.Module):
    """Swin Transformer.

    Args:
        strides: patch-embed / patch-merging stride per stage. ``strides[0]``
            must equal ``patch_size`` -- Swin's patches do not overlap.
        convert_weights: the checkpoint to load comes from the original
            Microsoft repo and needs ``swin_convert`` applied first.
        frozen_stages: stages (plus the patch embed) held in eval with grads
            off. ``-1`` freezes nothing.
    """

    def __init__(
        self,
        pretrain_img_size: int | tuple[int, int] = 224,
        in_channels: int = 3,
        embed_dims: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: int = 4,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        strides: tuple[int, ...] = (4, 2, 2, 2),
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_abs_pos_embed: bool = False,
        act_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        with_cp: bool = False,
        convert_weights: bool = False,
        frozen_stages: int = -1,
        init_cfg: dict | None = None,
    ):
        super().__init__()
        act_cfg = act_cfg or {"type": "GELU"}
        norm_cfg = norm_cfg or {"type": "LN"}
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        self.init_cfg = init_cfg

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif len(pretrain_img_size) == 1:
            pretrain_img_size = to_2tuple(pretrain_img_size[0])
        if len(pretrain_img_size) != 2:
            raise ValueError(f"pretrain_img_size must have length 1 or 2, got {pretrain_img_size}")

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        if strides[0] != patch_size:
            raise ValueError("strides[0] must equal patch_size (non-overlapping patch embed)")

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv2d",
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
        )

        if self.use_abs_pos_embed:
            num_patches = (pretrain_img_size[0] // patch_size) * (
                pretrain_img_size[1] // patch_size
            )
            self.absolute_pos_embed = nn.Parameter(torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Stochastic depth increases linearly with block index across the net.
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        stage_channels = embed_dims
        for i in range(num_layers):
            downsample = (
                PatchMerging(
                    in_channels=stage_channels,
                    out_channels=2 * stage_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                )
                if i < num_layers - 1
                else None
            )
            self.stages.append(
                SwinBlockSequence(
                    embed_dims=stage_channels,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * stage_channels,
                    depth=depths[i],
                    window_size=window_size,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=downsample,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                )
            )
            if downsample:
                stage_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        for i in out_indices:
            self.add_module(f"norm{i}", build_norm_layer(norm_cfg, self.num_features[i])[1])

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        return self

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f"norm{i - 1}")
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False
            stage = self.stages[i - 1]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def init_weights(self) -> None:
        """Random init, or load ``init_cfg['checkpoint']`` if one is given."""
        if self.init_cfg is None:
            if self.use_abs_pos_embed:
                nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
            return

        if "checkpoint" not in self.init_cfg:
            raise KeyError(f"{type(self).__name__} init_cfg must contain 'checkpoint'")

        from ..checkpoint import _extract_state_dict, _read_checkpoint

        state_dict = _extract_state_dict(_read_checkpoint(self.init_cfg["checkpoint"]))
        if self.convert_weights:
            state_dict = swin_convert(state_dict)
        state_dict = OrderedDict(
            (k[len("backbone.") :], v)
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        )
        if next(iter(state_dict), "").startswith("module."):
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

        self._adapt_pretrained(state_dict)
        self.load_state_dict(state_dict, strict=False)

    def _adapt_pretrained(self, state_dict: OrderedDict) -> None:
        """Reshape/interpolate the shape-dependent entries in place.

        The position-bias tables are indexed by window size, so a checkpoint
        trained at a different window size has to be resampled rather than
        dropped -- dropping it would quietly reset that table to zeros.
        """
        if state_dict.get("absolute_pos_embed") is not None:
            pos_embed = state_dict["absolute_pos_embed"]
            N1, L, C1 = pos_embed.size()
            N2, C2, H, W = self.absolute_pos_embed.size()
            if N1 == N2 and C1 == C2 and L == H * W:
                state_dict["absolute_pos_embed"] = (
                    pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
                )

        own = self.state_dict()
        for key in [k for k in state_dict if "relative_position_bias_table" in k]:
            table = state_dict[key]
            L1, nH1 = table.size()
            L2, nH2 = own[key].size()
            if nH1 != nH2 or L1 == L2:
                continue
            S1, S2 = int(L1**0.5), int(L2**0.5)
            resized = F.interpolate(
                table.permute(1, 0).reshape(1, nH1, S1, S1), size=(S2, S2), mode="bicubic"
            )
            state_dict[key] = resized.view(nH2, L2).permute(1, 0).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(out)
                out = (
                    out.view(-1, *out_hw_shape, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return outs


def swin_convert(ckpt: dict) -> OrderedDict:
    """Translate an original-repo Swin ``state_dict`` into this layout.

    Two kinds of change: renames (``layers``->``stages``, ``attn``->
    ``attn.w_msa``, ``mlp.fc1/fc2``->``ffn.layers.0.0/1``), and a channel
    *reordering* for patch merging. The original gathers the 2x2 neighbourhood
    as (top-left, bottom-left, top-right, bottom-right) while ``nn.Unfold``
    yields row-major order, so ``reduction.weight`` and ``norm`` need their
    four channel groups permuted.
    """
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        return x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        return x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)

    for key, value in ckpt.items():
        k = key[len("backbone.") :] if key.startswith("backbone") else key

        if k.startswith("head"):
            continue
        new_v = value
        if k.startswith("layers"):
            if "attn." in k:
                new_k = k.replace("attn.", "attn.w_msa.")
            elif "mlp." in k:
                if "mlp.fc1." in k:
                    new_k = k.replace("mlp.fc1.", "ffn.layers.0.0.")
                elif "mlp.fc2." in k:
                    new_k = k.replace("mlp.fc2.", "ffn.layers.1.")
                else:
                    new_k = k.replace("mlp.", "ffn.")
            elif "downsample" in k:
                new_k = k
                if "reduction." in k:
                    new_v = correct_unfold_reduction_order(value)
                elif "norm." in k:
                    new_v = correct_unfold_norm_order(value)
            else:
                new_k = k
            new_k = new_k.replace("layers", "stages", 1)
        elif k.startswith("patch_embed"):
            new_k = k.replace("proj", "projection") if "proj" in k else k
        else:
            new_k = k

        new_ckpt["backbone." + new_k] = new_v

    return new_ckpt

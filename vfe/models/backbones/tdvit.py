"""TDViT: Temporal Dilated Video Transformer (https://arxiv.org/abs/2402.09257).

A Swin Transformer in which some blocks attend across time. A *temporal dilated
transformer block* (TDTB) is a Swin block whose attention takes its queries
from the current frame and its keys and values from a reference map ``f^R`` of
an earlier frame (the paper's Eq. 1), through the block's own LayerNorm and
``qkv`` projection. It has no parameter a Swin block lacks, so TDViT-T loads
Swin-T's weights as they are and has Swin-T's parameter count (the paper's
Table 2). Each stage runs its Swin blocks (``'s'``) first and its TDTBs
(``'t'``) last -- the *split* scheme of Sec. 3.3; TDViT-T is
``('st', 'st', 'sssttt', 'st')``. The advanced variants (TDViT-T+, ...) add two
more TDTBs at the end of stage 3.

Where ``f^R`` comes from:

* **Inference, one frame at a time** (Sec. 3.2). Each TDTB keeps a memory of
  the maps it received for the last ``D_t`` frames of the video, ``D_t`` being
  its stage's temporal dilation (4 / 8 / 16 / 32 by default, Table 7). It
  samples ``f^R`` from that memory (the *temporal earliest* strategy by
  default, Table 8: the oldest map) and reuses it, keys and values included,
  for ``D_t`` frames before sampling again. A stage-``s`` TDTB therefore looks
  between ``D_t`` and ``2 D_t - 1`` frames back, and the stages together reach
  far further (Sec. 3.4). On a video's first frame the memory is empty and the
  frame attends to itself: the block is then exactly a Swin block.
* **Training** (Sec. 4). The memory is approximated by one reference frame per
  stage, sampled within ``±D_t`` of the key frame by the dataset
  (``ref_img_sampling(method='stagewise_uniform')``). Reference ``s`` passes,
  without gradients, through stages 1 to ``s``, every TDTB attending to the
  frame itself, and each TDTB of stage ``s`` keeps the map it received. The
  key frame then passes through the whole network, each TDTB attending to
  its kept map; only the key frame has losses.

What the paper leaves open follows the authors' code (the CVPR 2022
supplementary, built on mmdet 2.x): a TDTB remembers the map it *receives*
(its input; the paper's text says its output, and ``memory_feature='output'``
gives that), training references pass through the TDTBs as self-attention,
the shift of a block's windows alternates with its index in the stage as in
Swin, and extra TDTBs sit outside Swin's stochastic-depth schedule (drop path
0). The authors' ``MemoryQueue`` is not in that supplementary; the sampling
rules here are the paper's.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn

from ..builder import BACKBONES
from .swin import ShiftWindowMSA, SwinBlock, SwinTransformer, WindowMSA

__all__ = ["MemoryQueue", "SpatiotemporalSequence", "TDTB", "TDViT", "TemporalShiftWindowMSA",
           "TemporalWindowMSA"]


class MemoryQueue:
    """A TDTB's memory: its maps of the last ``max_length`` frames, oldest
    first, from which the reference map ``f^R`` is drawn.

    ``sample`` returns the reference for the current frame -- the frame itself
    while the memory is empty -- and draws a new one, by ``policy``, once the
    current one has served ``reuse`` frames. ``version`` changes with every
    draw, so a block can tell when its cached keys and values are stale.
    Random policies draw from torch's global CPU generator, which
    ``vfe.cli.test --seed`` seeds.

    Policies (the paper's Sec. 3.2 and Fig. 3):
        earliest: the oldest map.
        nms: the map with the largest L2 norm ("temporal NMS").
        patch_shuffle: the memory split into four groups of consecutive
            frames; the map's top-left, top-right, bottom-left and
            bottom-right quarters each come from a random frame of one group.
        channel_shuffle: each channel from a random frame.
    """

    POLICIES = ("earliest", "nms", "patch_shuffle", "channel_shuffle")

    def __init__(self, max_length: int, reuse: int | None = None, policy: str = "earliest"):
        if max_length < 1:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if policy not in self.POLICIES:
            raise ValueError(f"unknown memory sampling policy {policy!r}; known: {self.POLICIES}")
        self.max_length = max_length
        self.reuse = max_length if reuse is None else reuse
        if self.reuse < 1:
            raise ValueError(f"reuse must be positive, got {self.reuse}")
        self.policy = policy
        self.frames: deque[torch.Tensor] = deque(maxlen=max_length)
        self.reference: torch.Tensor | None = None
        self.version = 0
        self._age = 0

    def __len__(self) -> int:
        return len(self.frames)

    def reset(self) -> None:
        self.frames.clear()
        self.reference = None
        self._age = 0
        self.version += 1

    def update(self, x: torch.Tensor) -> None:
        """Remember the current frame's map; the oldest one falls out."""
        self.frames.append(x.detach())

    def sample(self, x: torch.Tensor, hw_shape: Sequence[int]) -> torch.Tensor:
        """``f^R`` for the current frame, whose map is ``x`` ``(B, L, C)``."""
        if not self.frames:
            return x
        if self.reference is None or self._age >= self.reuse:
            self.reference = self._draw(hw_shape)
            self.version += 1
            self._age = 0
        self._age += 1
        return self.reference

    def _draw(self, hw_shape: Sequence[int]) -> torch.Tensor:
        frames = list(self.frames)
        if self.policy == "earliest":
            return frames[0]
        if self.policy == "nms":
            norms = torch.stack([f.float().norm() for f in frames])
            return frames[int(norms.argmax())]
        stacked = torch.stack(frames)  # (T, B, L, C)
        T, B, L, C = stacked.shape
        if self.policy == "channel_shuffle":
            index = torch.randint(T, (C,)).to(stacked.device)
            return stacked.gather(0, index.view(1, 1, 1, C).expand(1, B, L, C))[0]
        # patch_shuffle
        H, W = hw_shape
        if H * W != L:
            raise ValueError(f"memory maps have {L} tokens but hw_shape {tuple(hw_shape)}")
        groups = [g for g in np.array_split(np.arange(T), 4)]
        picks = []
        for group in groups:
            pool = group if len(group) else np.arange(T)  # fewer than four frames so far
            picks.append(int(pool[int(torch.randint(len(pool), ()))]))
        grid = stacked.view(T, B, H, W, C)
        h, w = H // 2, W // 2
        top = torch.cat((grid[picks[0], :, :h, :w], grid[picks[1], :, :h, w:]), dim=2)
        bottom = torch.cat((grid[picks[2], :, h:, :w], grid[picks[3], :, h:, w:]), dim=2)
        return torch.cat((top, bottom), dim=1).reshape(B, L, C)


class TemporalWindowMSA(WindowMSA):
    """``WindowMSA`` that can also attend from one frame's windows to another's:
    queries from the first third of ``qkv``, keys and values from the rest.

    ``temporal_bias`` (None unless a TDTB enables it) is a learnable per-head
    logit added to the reference's keys in joint attention.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter("temporal_bias", None)

    def key_values(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head keys and values ``(nW*B, num_heads, N, head_dims)`` of
        normalised reference windows ``(nW*B, N, C)``."""
        B, N, C = windows.shape
        bias = self.qkv.bias[C:] if self.qkv.bias is not None else None
        kv = (F.linear(windows, self.qkv.weight[C:], bias)
              .reshape(B, N, 2, self.num_heads, C // self.num_heads)
              .permute(2, 0, 3, 1, 4))
        return kv[0], kv[1]

    def cross(self, x: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor],
              mask: torch.Tensor | None = None, joint: bool = False) -> torch.Tensor:
        """Attention of query windows ``(nW*B, N, C)`` over a reference's
        keys and values (from :meth:`key_values`, same windows) -- with
        ``joint``, over the windows' own keys and values too."""
        B, N, C = x.shape
        if joint:
            qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
            return self.attend_joint(qkv[0], torch.cat((qkv[1], kv[0]), dim=2),
                                     torch.cat((qkv[2], kv[1]), dim=2), mask)
        bias = self.qkv.bias[:C] if self.qkv.bias is not None else None
        q = (F.linear(x, self.qkv.weight[:C], bias)
             .reshape(B, N, self.num_heads, C // self.num_heads)
             .permute(0, 2, 1, 3))
        return self.attend(q, kv[0], kv[1], mask)

    def attend_joint(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attention of ``q`` ``(nW*B, num_heads, N, head_dims)`` over ``2N``
        keys and values: the window's own tokens, then the reference's. Both
        halves get the relative position bias (they share a grid) and the
        shift mask; the reference's, the temporal bias too."""
        B, num_heads, N, _ = q.shape
        C = num_heads * q.shape[3]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1)
        ref_bias = bias if self.temporal_bias is None else bias + self.temporal_bias.view(-1, 1, 1)
        if self.fused:
            return self.fused_attention(
                q, k, v, torch.cat((bias, ref_bias), dim=-1),
                None if mask is None else torch.cat((mask, mask), dim=-1))
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, heads, N, 2N)
        attn = attn + torch.cat((bias, ref_bias), dim=-1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B // nW, nW, num_heads, N, 2 * N)
                    + torch.cat((mask, mask), dim=-1).unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, num_heads, N, 2 * N)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TemporalShiftWindowMSA(ShiftWindowMSA):
    """Shifted-window attention whose keys and values may come from another
    frame's map: both maps are padded, shifted and cut into the same windows,
    so a query attends to the reference tokens of its own window, with the
    usual relative position bias and shift mask (the paper's window-based
    local attention, Fig. 4a)."""

    w_msa_cls = TemporalWindowMSA

    def forward(self, query: torch.Tensor, hw_shape: Sequence[int],
                kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                joint: bool = False) -> torch.Tensor:
        """``kv`` from :meth:`key_values`; without it, self-attention as in Swin.
        ``joint`` attends over the query's own window too."""
        if kv is None:
            return super().forward(query, hw_shape)
        B, L, C = query.shape
        H, W = hw_shape
        windows, (H_pad, W_pad) = self.to_windows(query, hw_shape)
        attn_mask = (self.shift_attn_mask(H_pad, W_pad, query.device)
                     if self.shift_size > 0 else None)
        attn_windows = self.w_msa.cross(windows, kv, mask=attn_mask, joint=joint)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, H_pad, W_pad)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if H_pad > H or W_pad > W:
            x = x[:, :H, :W, :].contiguous()
        return self.drop(x.view(B, H * W, C))

    def key_values(self, key: torch.Tensor,
                   hw_shape: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values of a normalised reference map ``(B, L, C)``, in this
        block's (shifted) windows."""
        windows, _ = self.to_windows(key, hw_shape)
        return self.w_msa.key_values(windows)

    def to_windows(self, x: torch.Tensor,
                   hw_shape: Sequence[int]) -> tuple[torch.Tensor, tuple[int, int]]:
        """``(B, L, C)`` tokens -> padded, shifted windows ``(nW*B, Wh*Ww, C)``,
        and the padded map's size."""
        B, L, C = x.shape
        H, W = hw_shape
        if L != H * W:
            raise ValueError(f"input has {L} tokens but hw_shape {tuple(hw_shape)} implies {H * W}")
        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        return self.window_partition(x).view(-1, self.window_size**2, C), (H_pad, W_pad)


class TDTB(SwinBlock):
    """Temporal dilated transformer block (the paper's Eq. 1 and Fig. 2b):

        f^R = Sampling(M, D_t)
        f'  = MCA(LN(f), LN(f^R)) + f
        out = MLP(LN(f')) + f'

    with the parameters of a :class:`SwinBlock`: ``norm1`` normalises both maps
    and ``qkv`` projects the queries of one and the keys and values of the
    other. Without a reference it is a Swin block.

    Args:
        temporal_dilation: ``D_t``: the memory's length, and how many frames
            a sampled reference serves.
        memory_sampling: :class:`MemoryQueue`'s policy.
        memory_feature: what the memory keeps of each frame, the block's
            ``'input'`` (the authors' code) or its ``'output'`` (the text).
        memory_reuse: frames a reference serves; None for ``D_t``.
        attention: ``'cross'``, the paper's: the keys and values are the
            reference's alone, so the block gives up the frame's own spatial
            attention. ``'joint'``: the frame's window and the reference's
            together (a two-frame space-time window), so a query can keep to
            its own frame where the reference does not match; with the frame
            as its own reference this is exactly the Swin block too.
        temporal_bias: with ``'joint'``, a learnable per-head logit (from 0)
            on the reference's keys -- how far each head trusts it.
        Other arguments are :class:`SwinBlock`'s.
    """

    attn_cls = TemporalShiftWindowMSA

    def __init__(self, *args, temporal_dilation: int = 1, memory_sampling: str = "earliest",
                 memory_feature: str = "input", memory_reuse: int | None = None,
                 attention: str = "cross", temporal_bias: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if memory_feature not in ("input", "output"):
            raise ValueError(f"memory_feature must be 'input' or 'output', got {memory_feature!r}")
        if attention not in ("cross", "joint"):
            raise ValueError(f"attention must be 'cross' or 'joint', got {attention!r}")
        if temporal_bias and attention != "joint":
            raise ValueError("temporal_bias needs attention='joint'")
        self.temporal_dilation = temporal_dilation
        self.memory_feature = memory_feature
        self.joint = attention == "joint"
        if temporal_bias:
            self.attn.w_msa.temporal_bias = nn.Parameter(torch.zeros(self.attn.w_msa.num_heads))
        # Per-video inference state, never part of a checkpoint.
        self.memory = MemoryQueue(temporal_dilation, memory_reuse, memory_sampling)
        self._kv_cache: tuple | None = None

    def forward(self, x: torch.Tensor, hw_shape: Sequence[int], ref: torch.Tensor | None = None,
                kv: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """``x`` attends to a reference map ``ref`` ``(B, L, C)``, to
        precomputed keys and values ``kv`` (:meth:`key_values`), or with
        neither to itself."""

        def _inner_forward(x, ref):
            identity = x
            x = self.norm1(x)
            if ref is not None:
                x = self.attn(x, hw_shape, kv=self.key_values(ref, hw_shape), joint=self.joint)
            else:
                x = self.attn(x, hw_shape, kv=kv, joint=self.joint)
            x = x + identity

            identity = x
            x = self.norm2(x)
            return self.ffn(x, identity=identity)

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x, ref, use_reentrant=False)
        return _inner_forward(x, ref)

    def key_values(self, ref: torch.Tensor,
                   hw_shape: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """The keys and values a reference map ``(B, L, C)`` offers."""
        return self.attn.key_values(self.norm1(ref), hw_shape)

    def forward_online(self, x: torch.Tensor, hw_shape: Sequence[int]) -> torch.Tensor:
        """The current frame of a video seen in order: attend to the memory's
        reference, then remember this frame. A reference's keys and values are
        computed once and reused while it serves (unless autograd is on)."""
        ref = self.memory.sample(x, hw_shape)
        if ref is x:  # the video's first frame
            out = self(x, hw_shape)
        elif torch.is_grad_enabled():
            out = self(x, hw_shape, ref=ref)
        else:
            stamp = (self.memory.version, tuple(hw_shape))
            if self._kv_cache is None or self._kv_cache[0] != stamp:
                self._kv_cache = (stamp, self.key_values(ref, hw_shape))
            out = self(x, hw_shape, kv=self._kv_cache[1])
        self.memory.update(x if self.memory_feature == "input" else out)
        return out

    def reset_memory(self) -> None:
        self.memory.reset()
        self._kv_cache = None


class SpatiotemporalSequence(nn.Module):
    """One TDViT stage: Swin blocks and TDTBs in ``layout`` order (``'s'`` and
    ``'t'``), then ``extra_tdtbs`` more TDTBs, then the downsample.

    As in Swin, block ``i`` shifts its windows when ``i`` is odd. The extra
    TDTBs (the advanced variants') take no part in stochastic depth, as in the
    authors' code. Other arguments are :class:`SwinBlockSequence`'s and
    :class:`TDTB`'s.
    """

    def __init__(self, embed_dims: int, num_heads: int, feedforward_channels: int, depth: int,
                 layout: str, temporal_dilation: int, extra_tdtbs: int = 0,
                 memory_sampling: str = "earliest", memory_feature: str = "input",
                 memory_reuse: int | None = None, attention: str = "cross",
                 temporal_bias: bool = False, window_size: int = 7, qkv_bias: bool = True,
                 qk_scale: float | None = None, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float | list[float] = 0.0,
                 downsample: nn.Module | None = None, act_cfg: dict | None = None,
                 norm_cfg: dict | None = None, with_cp: bool = False):
        super().__init__()
        if len(layout) != depth or set(layout) - {"s", "t"}:
            raise ValueError(f"layout {layout!r} must be {depth} of 's' and 't'")
        if isinstance(drop_path_rate, list):
            if len(drop_path_rate) != depth:
                raise ValueError(f"expected {depth} drop path rates, got {len(drop_path_rate)}")
            drop_path_rates = list(drop_path_rate)
        else:
            drop_path_rates = [drop_path_rate] * depth
        drop_path_rates += [0.0] * extra_tdtbs
        self.layout = layout + "t" * extra_tdtbs

        self.blocks = nn.ModuleList()
        for i, kind in enumerate(self.layout):
            block_cfg = dict(
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
            if kind == "s":
                self.blocks.append(SwinBlock(**block_cfg))
            else:
                self.blocks.append(TDTB(temporal_dilation=temporal_dilation,
                                        memory_sampling=memory_sampling,
                                        memory_feature=memory_feature,
                                        memory_reuse=memory_reuse, attention=attention,
                                        temporal_bias=temporal_bias, **block_cfg))
        self.downsample = downsample

    def forward(self, x: torch.Tensor, hw_shape, refs: list | None = None, online: bool = False):
        """With ``refs`` (one map per block, None for Swin blocks) each TDTB
        attends to its own; ``online``, to its memory; else to the frame."""
        for j, block in enumerate(self.blocks):
            if not isinstance(block, TDTB):
                x = block(x, hw_shape)
            elif refs is not None:
                x = block(x, hw_shape, ref=refs[j])
            elif online:
                x = block.forward_online(x, hw_shape)
            else:
                x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        return x, hw_shape, x, hw_shape


@BACKBONES.register_module()
class TDViT(SwinTransformer):
    """Temporal Dilated Video Transformer.

    ``forward(img, ref_img)`` trains: ``ref_img`` ``(B, num_stages, 3, H, W)``
    holds each key frame's references, one per stage, in stage order. In eval
    mode ``forward(img)`` is online inference: frames of one video in order,
    one call each, :meth:`reset_memory` before a video's first.

    Args:
        layout: per stage, its blocks as ``'s'`` (Swin) and ``'t'`` (TDTB);
            the stages' depths. TDViT-T: ``('st', 'st', 'sssttt', 'st')``.
        extra_tdtbs: per stage, TDTBs appended after ``layout``'s blocks
            (TDViT-T+: ``(0, 0, 2, 0)``); they have no pretrained weights.
        extra_init: how :meth:`init_weights` starts the extra TDTBs:
            ``'default'``, torch's initialisation (the authors' code);
            ``'zero'``, the output projections of both residual branches at
            zero, so each extra block starts as the identity and the network
            as TDViT without them.
        temporal_dilations: ``D_t`` per stage.
        memory_sampling, memory_feature, memory_reuse, attention,
        temporal_bias: :class:`TDTB`'s.
        Other arguments are :class:`~vfe.models.backbones.SwinTransformer`'s;
        ``depths``, if given, must match ``layout``.
    """

    def __init__(self, layout: Sequence[str] = ("st", "st", "sssttt", "st"),
                 extra_tdtbs: Sequence[int] | None = None, extra_init: str = "default",
                 temporal_dilations: Sequence[int] = (4, 8, 16, 32),
                 memory_sampling: str = "earliest", memory_feature: str = "input",
                 memory_reuse: int | None = None, attention: str = "cross",
                 temporal_bias: bool = False, depths: Sequence[int] | None = None,
                 **kwargs):
        if extra_init not in ("default", "zero"):
            raise ValueError(f"extra_init must be 'default' or 'zero', got {extra_init!r}")
        num_stages = len(layout)
        extra_tdtbs = tuple(extra_tdtbs) if extra_tdtbs is not None else (0,) * num_stages
        if len(temporal_dilations) != num_stages or len(extra_tdtbs) != num_stages:
            raise ValueError(f"layout has {num_stages} stages; temporal_dilations and "
                             "extra_tdtbs need one entry per stage")
        layout_depths = tuple(len(stage) for stage in layout)
        if depths is not None and tuple(depths) != layout_depths:
            raise ValueError(f"depths {tuple(depths)} do not match layout {tuple(layout)}")
        stage_cfgs = [
            dict(layout=layout[i], temporal_dilation=temporal_dilations[i],
                 extra_tdtbs=extra_tdtbs[i], memory_sampling=memory_sampling,
                 memory_feature=memory_feature, memory_reuse=memory_reuse,
                 attention=attention, temporal_bias=temporal_bias)
            for i in range(num_stages)
        ]
        super().__init__(depths=layout_depths, stage_cfgs=stage_cfgs, **kwargs)
        self.temporal_dilations = tuple(temporal_dilations)
        self.memory_feature = memory_feature
        self.extra_tdtbs = extra_tdtbs
        self.extra_init = extra_init

    def make_stage(self, **kwargs) -> nn.Module:
        return SpatiotemporalSequence(**kwargs)

    def init_weights(self) -> None:
        super().init_weights()
        if self.extra_init != "zero":
            return
        for stage, num_extra in zip(self.stages, self.extra_tdtbs, strict=True):
            for block in stage.blocks[len(stage.blocks) - num_extra:]:
                for linear in (block.attn.w_msa.proj, block.ffn.layers[1]):
                    nn.init.zeros_(linear.weight)
                    nn.init.zeros_(linear.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.reset_memory()
        return self

    def reset_memory(self) -> None:
        """Forget the current video; the next frame is treated as its first."""
        for module in self.modules():
            if isinstance(module, TDTB):
                module.reset_memory()

    def _embed(self, img: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x, hw_shape = self.patch_embed(img)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        return self.drop_after_pos(x), hw_shape

    @torch.no_grad()
    def reference_maps(self, ref_img: torch.Tensor) -> list[list[torch.Tensor | None]]:
        """Per stage and block, the map each TDTB keeps from its stage's
        reference (None for Swin blocks); ``ref_img`` is
        ``(B, num_stages, 3, H, W)``.

        All references enter stage 1; stage ``s``'s own reference leaves the
        batch once its last TDTB has what it keeps. TDTBs attend to the frame
        itself here, as in the authors' code.
        """
        B, num_refs = ref_img.shape[:2]
        if num_refs != len(self.stages):
            raise ValueError(f"TDViT needs one reference per stage ({len(self.stages)}), "
                             f"got {num_refs}")
        # Stage-major: the first B rows are always the current stage's references.
        x, hw_shape = self._embed(ref_img.transpose(0, 1).flatten(0, 1))
        keep_input = self.memory_feature == "input"
        maps: list[list[torch.Tensor | None]] = []
        for stage in self.stages:
            tdtbs = [j for j, block in enumerate(stage.blocks) if isinstance(block, TDTB)]
            stage_maps: list[torch.Tensor | None] = [None] * len(stage.blocks)
            for j, block in enumerate(stage.blocks):
                if keep_input and j in tdtbs:
                    stage_maps[j] = x[:B]
                    if j == tdtbs[-1]:
                        x = x[B:]
                if not len(x):  # the last stage, past its last TDTB
                    break
                x = block(x, hw_shape)
                if not keep_input and j in tdtbs:
                    stage_maps[j] = x[:B]
                    if j == tdtbs[-1]:
                        x = x[B:]
            if not tdtbs:
                x = x[B:]
            maps.append(stage_maps)
            if stage.downsample is not None and len(x):
                x, hw_shape = stage.downsample(x, hw_shape)
        return maps

    def forward(self, x: torch.Tensor, ref_img: torch.Tensor | None = None) -> list[torch.Tensor]:
        if ref_img is None and self.training:
            raise ValueError("TDViT trains with reference frames, one per stage")
        refs = self.reference_maps(ref_img) if ref_img is not None else None
        return self._run(x, refs, online=refs is None)

    def forward_spatial(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Frames on their own, every TDTB attending to its frame (a Swin
        network), the memories untouched: SELSA's reference frames."""
        return self._run(x, None, online=False)

    def _run(self, x: torch.Tensor, refs: list | None, online: bool) -> list[torch.Tensor]:
        x, hw_shape = self._embed(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(
                x, hw_shape, refs=refs[i] if refs is not None else None, online=online)
            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(out)
                out = (
                    out.view(-1, *out_hw_shape, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return outs

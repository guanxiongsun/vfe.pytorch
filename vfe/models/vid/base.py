"""Base class for video detectors. Port of ``mmdet.models.vid.base``.

A video detector wraps a still-image detector and adds reference frames:
``forward_train`` sees a key frame plus sampled references, ``simple_test``
sees one frame at a time in video order and keeps whatever cross-frame state
the method needs. mmcv runner hooks, TTA and ``show_result`` are dropped, as
in :mod:`vfe.models.detectors.base`; losses go through
:func:`vfe.models.detectors.parse_losses`.
"""

from __future__ import annotations

from torch import nn

__all__ = ["BaseVideoDetector"]


class BaseVideoDetector(nn.Module):
    def freeze_module(self, module: str | list[str] | tuple[str, ...]) -> None:
        """Put the named submodule(s) in eval mode and stop their gradients.

        Note ``eval()`` does not survive a later ``model.train()``; the
        original has the same limitation.
        """
        if isinstance(module, str):
            modules = [module]
        elif isinstance(module, (list, tuple)):
            modules = module
        else:
            raise TypeError("module must be a str or a list")
        for name in modules:
            m = getattr(self, name)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @property
    def with_detector(self) -> bool:
        return getattr(self, "detector", None) is not None

    def init_weights(self) -> None:
        self.detector.init_weights()

    def forward_train(self, img, img_metas, ref_img=None, ref_img_metas=None, **kwargs):
        raise NotImplementedError

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def forward(self, img, img_metas, ref_img=None, ref_img_metas=None, return_loss=True,
                **kwargs):
        """Losses when ``return_loss``, else detections for one frame.

        For inference, ``img`` / ``img_metas`` may carry mmdet's extra
        test-time-augmentation nesting (a list with one entry); more than one
        augmentation raises. ``ref_img`` / ``ref_img_metas`` are passed through
        untouched -- ``simple_test`` unwraps them itself.
        """
        if return_loss:
            return self.forward_train(
                img, img_metas, ref_img=ref_img, ref_img_metas=ref_img_metas, **kwargs
            )
        if isinstance(img, list):
            if len(img) != 1:
                raise NotImplementedError(
                    f"test-time augmentation is not ported; got {len(img)} augmentations"
                )
            img = img[0]
        if isinstance(img_metas[0], list):
            if len(img_metas) != 1:
                raise NotImplementedError("test-time augmentation is not ported")
            img_metas = img_metas[0]
        if "proposals" in kwargs:
            kwargs["proposals"] = kwargs["proposals"][0]
        return self.simple_test(
            img, img_metas, ref_img=ref_img, ref_img_metas=ref_img_metas, **kwargs
        )

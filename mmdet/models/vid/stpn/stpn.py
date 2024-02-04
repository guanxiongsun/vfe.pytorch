import warnings
import torch
from mmdet.models import build_detector
from ...builder import MODELS
from ..base import BaseVideoDetector

from .dvp_predictor import AttentionPredictor, AveragePredictor


@MODELS.register_module()
class STPN(BaseVideoDetector):
    """Spation-temporal Prompting Network (STPN) for Video Object Detection.
    TODO: change url <https://arxiv.org/abs/1907.06390>
    """

    def __init__(self,
                 detector,
                 pretrained=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None,
                 predictor='att',
                 embed_dims=768,
                 num_prompts=5,
                 prompt_dims=96,
                 ):
        super(STPN, self).__init__(init_cfg)
        if isinstance(pretrained, dict):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            detector_pretrain = pretrained.get('detector', None)
            if detector_pretrain:
                detector.init_cfg = dict(
                    type='Pretrained', checkpoint=detector_pretrain)
            else:
                detector.init_cfg = None
        self.detector = build_detector(detector)
        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # create prompt predict network
        print("The predictor is {}".format(predictor))
        if predictor == 'att':
            self.prompt_predictor = AttentionPredictor(embed_dims,
                                                       num_prompts=num_prompts,
                                                       prompt_dims=prompt_dims)
        elif predictor == 'avg':
            self.prompt_predictor = AveragePredictor(embed_dims,
                                                     num_prompts=num_prompts,
                                                     prompt_dims=prompt_dims,
                                                     )
        else:
            raise ValueError

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img,
                    # #   ref_img_metas,
                    #   ref_gt_bboxes,
                    #   ref_gt_labels,
                    #   gt_instance_ids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                    #   ref_gt_instance_ids=None,
                    #   ref_gt_bboxes_ignore=None,
                    #   ref_gt_masks=None,
                    #   ref_proposals=None,
                      **kwargs):
        assert len(img) == 1, \
            'selsa video detector only supports 1 batch size per gpu for now.'

        # [B, C, H, W]
        with torch.no_grad():
            ref_x = self.detector.backbone(ref_img[0])[-1]

        # [num_prompt, C]
        prompt = self.prompt_predictor(ref_x)

        x = self.detector.backbone(img, prompt)
        if self.detector.with_neck:
            x = self.detector.neck(x)

        losses = dict()
        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

        else:
            proposal_list = proposals

        roi_losses = self.detector.roi_head.forward_train(x, img_metas, proposal_list,
                                                          gt_bboxes, gt_labels,
                                                          gt_bboxes_ignore, gt_masks,
                                                          **kwargs)
        losses.update(roi_losses)

        return losses

    def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
        """Extract features for `img` during testing.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (Tensor | None): of shape (1, N, C, H, W) encoding input
                reference images. Typically these should be mean centered and
                std scaled. N denotes the number of reference images. There
                may be no reference images in some cases.

            ref_img_metas (list[list[dict]] | None): The first list only has
                one element. The second list contains image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.

        Returns:
            tuple(x, img_metas, ref_x, ref_img_metas): x is the multi level
                feature maps of `img`, ref_x is the multi level feature maps
                of `ref_img`.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        frame_stride = img_metas[0].get('frame_stride', -1)

        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                # [B, C, H, W]
                ref_x = self.detector.backbone(ref_img[0])[-1]
                # # [B*K, C]
                # B, C, H, W = ref_x.shape
                # ref_x = ref_x.view(B, C, -1).permute(0, 2, 1)
                # ref_x = self.get_topk(ref_x)
                # [num_prompt, C]
                self.prompt = self.prompt_predictor(ref_x)

            x = self.detector.backbone(img, self.prompt)
            if self.detector.with_neck:
                x = self.detector.neck(x)

        return x, img_metas

    def simple_test(self,
                    img,
                    img_metas,
                    ref_img=None,
                    ref_img_metas=None,
                    proposals=None,
                    ref_proposals=None,
                    rescale=False):
        if ref_img is not None:
            ref_img = ref_img[0]
        if ref_img_metas is not None:
            ref_img_metas = ref_img_metas[0]
        x, img_metas = self.extract_feats(img, img_metas, ref_img, ref_img_metas)

        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
        else:
            proposal_list = proposals

        outs = self.detector.roi_head.simple_test(
            x,
            proposal_list,
            img_metas,
            rescale=rescale)

        # results = dict()
        # results['det_bboxes'] = outs[0]
        # if len(outs) == 2:
        #     results['det_masks'] = outs[1]

        results = outs
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError

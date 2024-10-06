from cmath import log
from concurrent.futures import process
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from ssd.modeling import registry
from ssd.modeling.anchors.prior_box import PriorBox
from ssd.modeling.box_head.box_predictor import BoxPredictor, make_box_predictor
from ssd.modeling.box_head.roi_align import SSDRPN
from ssd.modeling.heatmap_head.upsampling import UpSampling
from ssd.utils import box_utils, det_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss, MultiBoxLossWithNeg


@registry.BOX_HEADS.register('SSDBoxHead')
class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLossWithNeg(
            neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(
            cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}


@registry.BOX_HEADS.register('SSDBoxRPNHead')
class SSDBoxRPNHead(SSDBoxHead):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.roi_pool = SSDRPN(cfg)
        self.up_sample_blocks = nn.ModuleList([
            UpSampling(512, 256),
            UpSampling(256, 128),
            UpSampling(128, 1),
        ])

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets, features)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets, features=None):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(
            cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )

        detections = (cls_logits, bbox_pred)

        if features is not None:
            features_dict = {str(k): v for k, v in enumerate(features)}
            decoding_results = self.decode_boxes(detections)
            proposals, labels = decoding_results
            object_feat = self.roi_pool(features_dict, proposals)
            for m in self.up_sample_blocks:
                object_feat = m(object_feat)

            print(object_feat.shape)

        return detections, loss_dict

    def decode_boxes(self, detections):
        cls_logits, bbox_pred = detections
        pred_results, _ = self._forward_test(cls_logits, bbox_pred)
        proposals = []
        labels = []
        for i, result in enumerate(pred_results):
            keep = result['scores'] > 0.5
            proposals.append(result['boxes'][keep])
            labels.append(result['labels'][keep])

        decoding_results = (proposals, labels)

        return decoding_results

    def _forward_test(self, cls_logits, bbox_pred):
        # np_cls_logits = cls_logits.cpu().detach().numpy()
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        # scores = cls_logits.sigmoid()
        np_scores = scores.cpu().detach().numpy()
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}


@registry.BOX_HEADS.register('SSDLSTMHead')
class SSDLSTMHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(
            neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(
            cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}


@registry.BOX_HEADS.register('SSDRPNHead')
class SSDRPNHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(
            neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.proposal_matcher = det_utils.Matcher()
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(100, 0.25)
        self.pooler = SSDRPN()
        self.priors = None
        self.cls_head = nn.Linear(128 * 128, 4)

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将gt_boxes拼接到proposal后面
        Args:
            proposals: 一个batch中每张图像rpn预测的boxes
            gt_boxes:  一个batch中每张图像对应的真实目标边界框

        Returns:

        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def assign_gt_to_proposals(self,
                               proposals,
                               orig_targets):
        """
        为每个proposal匹配对应的gt_box,并划分到正负样本中
        proposal only use for computing ious
        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        gt_boxes = []
        gt_labels = []
        for gt in orig_targets:
            gt_boxes.append(torch.from_numpy(gt["boxes"] * 300.0).to(proposals[0].device))
            gt_labels.append(torch.from_numpy(gt["labels"]).to(proposals[0].device))

        proposals = self.add_gt_proposals(proposals, gt_boxes)

        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算proposal与每个gt_box的iou重合度
                match_quality_matrix = det_utils.box_iou(gt_boxes_in_image, proposals_in_image)

                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1， low_threshold <= iou < high_threshold索引值为 -2
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # 获取proposal匹配到的gt对应标签
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                # 将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                # 将gt索引为-2的类别设置为-1, 即废弃样本
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLD  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return proposals, matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引（包括正样本和负样本）
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    # def select_gts(self,
    #                sampled_idx,
    #                proposa
    #                gt_boxes):

    def postprocess_wo_classes(self,
                               bbox_pred,
                               cls_logits):
        nms_indices = []
        nms_boxes = []
        bg_masks = []
        nms_cls_logits = []
        num_classes = self.cfg.MODEL.NUM_CLASSES
        regression_pred = []
        for box, box_pred, logit in zip(bbox_pred, bbox_pred, cls_logits):
            scores = F.softmax(logit, dim=1)
            conf, labels = torch.max(scores, dim=1)

            if self.priors is None:
                self.priors = PriorBox(self.cfg)().to(bbox_pred.device)

            box = box_utils.convert_locations_to_boxes(
                box, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
            )
            box = box_utils.center_form_to_corner_form(box)

            # mask = labels > 0
            # fg_boxes, fg_labels, fg_conf, fg_box_pred = box[mask], labels[mask], conf[mask], box_pred[mask]

            img_size = self.cfg.INPUT.IMAGE_SIZE
            box[:, 0::2] *= img_size
            box[:, 1::2] *= img_size

            nms_index = torchvision.ops.batched_nms(
                box, conf, labels, 0.7)[:100]
            nms_indices.append(nms_index)

            box = box[nms_index]
            box_pred = box_pred[nms_index]
            # logit = logit[nms_index]

            nms_boxes.append(box)
            # bg_masks.append(mask)
            regression_pred.append(box_pred)
            # nms_cls_logits.append(logit)

        return nms_indices, bg_masks, nms_boxes

    def _encode_boxes(self, 
                      bboxes: torch.Tensor, 
                      is_center_form: bool = False):
        priors = PriorBox(self.cfg)().to(bboxes.device)  # xywh format
        xyxy_priors = box_utils.center_form_to_corner_form(priors)  # xyxy format

        # if input is in xywh format
        if is_center_form:
            bboxes = box_utils.center_form_to_corner_form(bboxes)

        # input boxes should be xyxy format
        ious = box_utils.iou_of(bboxes.unsqueeze(0), xyxy_priors.unsqueeze(1))

        _, idx = ious.max(0)
        proposal_priors = priors[idx]

        xywh_bboxes = box_utils.corner_form_to_center_form(bboxes)

        # should be in xywh format
        encoded_bboxes = box_utils.convert_boxes_to_locations(xywh_bboxes, proposal_priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE)  # center_variance, corner_variance

        return encoded_bboxes

    def encode(self,
               proposals: torch.Tensor,
               gt_boxes: torch.Tensor):
        encoded_proposals = []
        encoded_gt_boxes = []

        for proposal, gt_box in zip(proposals, gt_boxes):
            encoded_proposal = self._encode_boxes(proposal / 300.0)
            encoded_gt_box = self._encode_boxes(gt_box)
            encoded_proposals.append(encoded_proposal)
            encoded_gt_boxes.append(encoded_gt_box)

        return torch.stack(encoded_proposals), torch.stack(encoded_gt_boxes)

    def forward(self, features, targets=None, orig_targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(features, cls_logits, bbox_pred, targets, orig_targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, features, cls_logits, bbox_pred, targets, orig_targets=None):
        device = bbox_pred.device
        # gt_boxes, gt_labels = targets['boxes'], targets['labels']

        # with torch.no_grad():
        nms_indices, bg_masks, processed_boxes = self.postprocess_wo_classes(
            bbox_pred=bbox_pred, cls_logits=cls_logits)

        features_dict = {}
        for i, feature in enumerate(features):
            features_dict[str(i)] = feature

        proposals, matched_idxs, labels = self.assign_gt_to_proposals(processed_boxes, orig_targets)
        sampled_idx = self.subsample(labels)
        matched_gt_boxes = []

        gt_boxes = [torch.from_numpy(target['boxes']) for target in orig_targets]

        for idx in range(len(proposals)):
            img_sampled_idx = sampled_idx[idx]
            proposals[idx] = proposals[idx][img_sampled_idx]
            labels[idx] = labels[idx][img_sampled_idx]
            matched_idxs[idx] = matched_idxs[idx][img_sampled_idx]

            # gt_boxes_in_image = torch.from_numpy(orig_targets[idx]['boxes'])
            gt_boxes_in_image = gt_boxes[idx]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), device=processed_boxes[0].device)

            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[idx]])

        labels = torch.stack(labels).to(device)
        matched_gt_boxes = torch.stack(matched_gt_boxes).to(device)

        object_feat = self.pooler(features_dict, processed_boxes)
        object_feat = object_feat.flatten(2)
        out = self.trans(object_feat)
        out = out.flatten(1)
        fc_output = self.cls_head(out).reshape(16, 100, -1)

        nms_boxes = torch.stack(processed_boxes)
        bbox_result, gt_bboxes = self.encode(nms_boxes, matched_gt_boxes)
        # nms_cls_logits = torch.stack(processed_logits)
        # boxes, labels = self.assign_gt_to_proposals(bbox_pred, nms_indices, gt_labels, gt_boxes, 0.5)

        # nms_gt_labels, nms_gt_boxes = self.select_gts(
        #     nms_indices, bg_masks, gt_labels, gt_boxes)
        # assert torch.any(gt_labels)

        reg_loss, cls_loss = self.loss_evaluator(
            fc_output, bbox_result, labels, gt_bboxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections, _ = self.post_processor(detections)
        return detections, {}

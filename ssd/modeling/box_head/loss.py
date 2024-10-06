import torch.nn as nn
import torch.nn.functional as F
import torch

from ssd.utils import box_utils


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, cls_labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, cls_labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), cls_labels[mask], reduction='sum')

        pos_mask = cls_labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class MultiBoxLossWithNeg(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLossWithNeg, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """

        num_classes = confidence.size(2)
        smooth_l1_loss = torch.tensor(0, device=labels.device, dtype=torch.float32)
        classification_loss = torch.tensor(0, device=labels.device, dtype=torch.float32)
        num_pos_samples = 0.
        num_neg_samples = 0.

        for pred_conf, pred_bboxes, gt_label, gt_bboxes in zip(confidence, predicted_locations, labels, gt_locations):
            if not torch.any(gt_label):
                with torch.no_grad():
                    # derived from cross_entropy=sum(log(p))
                    loss = -F.log_softmax(pred_conf, dim=1)[:, 0]
                    _, indexes = loss.sort(dim=0, descending=True)
                    _, orders = indexes.sort(dim=0)
                    mask = orders < torch.tensor(self.neg_pos_ratio)
                pred_conf = pred_conf[mask, :]
                gt_label = gt_label[mask]
                classification_loss += F.cross_entropy(pred_conf.view(-1, num_classes),
                                                       gt_label, reduction='sum')
                num_neg_samples += 3.
            else:
                with torch.no_grad():
                    # derived from cross_entropy=sum(log(p))
                    loss = -F.log_softmax(pred_conf, dim=1)[:, 0]
                    mask = box_utils.hard_negative_mining(loss, gt_label, self.neg_pos_ratio)

                pred_conf = pred_conf[mask, :]

                pos_mask = gt_label > 0
                pred_bboxes = pred_bboxes[pos_mask, :].view(-1, 4)
                gt_bboxes = gt_bboxes[pos_mask, :].view(-1, 4)
                smooth_l1_loss += F.smooth_l1_loss(pred_bboxes, gt_bboxes,
                                                   reduction='sum')

                classification_loss += F.cross_entropy(pred_conf.view(-1, num_classes), gt_label[mask], reduction='sum')
                num_pos_samples += gt_bboxes.shape[0]

        return smooth_l1_loss / max(num_pos_samples, 1), classification_loss / max((num_neg_samples + num_pos_samples), 1)

import torchvision.ops
from numpy import dtype
import torch

from ssd.structures.container import Container
from ssd.utils.nms import batched_nms


class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE
        self.height = cfg.INPUT.IMAGE_SIZE

    def __call__(self, detections):
        keeps = []
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # indices = torch.nonzero(scores > self.cfg.TEST.CONFIDENCE_THRESHOLD).squeeze(1)

            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height

            boxes = torch.where(torch.isinf(boxes), torch.full_like(boxes, 1.0), boxes)
            # indices = torch.nonzero(scores > self.cfg.TEST.CONFIDENCE_THRESHOLD).squeeze(1)
            # high_score_idxs = torch.gt(scores, self.cfg.TEST.CONFIDENCE_THRESHOLD)
            # temp = []
            # # indices = [i for i in range(len(high_score_idxs)) if high_score_idxs[i]]
            # for i in range(len(high_score_idxs)):
            #     if high_score_idxs[i]:
            #         temp.append(i)
            # indices = torch.tensor(temp)
            # # indices = scores[high_score_idxs].squeeze(1)

            # # if torch.any(high_score_idxs):
            # #     indices = scores[high_score_idxs].squeeze(1)
            # # else:
            # #     indices = torch.Tensor([],dtype=torch.int64)

            # boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            # keep = torchvision.ops.batched_nms(boxes, scores, labels, 0.5)
            keep = torchvision.ops.batched_nms(boxes, scores, labels, 0.2)
            # keep = batched_nms(boxes, scores, labels, self.cfg.TEST.NMS_THRESHOLD)
            # # # keep only topk scoring predictions
            keep = keep[:self.cfg.TEST.MAX_PER_IMAGE]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
            keeps.append(keep)
        return results

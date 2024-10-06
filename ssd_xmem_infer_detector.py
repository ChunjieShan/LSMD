import math
from ssd.config import cfg
import numpy as np

import torch
import cv2 as cv
import os
import pandas as pd
from ssd.modeling.detector.ssd_detector import SSDLSTPANDetector
from ssd.data.transforms import build_transforms
from typing import List
from vizer.draw import draw_boxes
from ssd.data.datasets.dark_xmem_lst_vid import DarkVIDDataset
from ssd.config.path_catlog import DatasetCatalog
from ssd.structures.container import Container


class SSDXMemDetector(SSDLSTPANDetector):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, curr_video_clip, memory_buffer=None, targets=None):
        if memory_buffer is None:
            curr_feat = self.forward_backbone_attn(curr_video_clip)
            curr_xs = curr_feat["dark5"][0].squeeze(2)
            x_lt, memory_long = curr_xs, curr_xs
            curr_feat["dark5"] = curr_xs
            neck_outputs = self.backbone.forward_pan(curr_feat)
        else:
            neck_outputs, x_lt, memory_long = self.forward_infer(curr_video_clip, memory_buffer)

        if isinstance(neck_outputs, torch.Tensor):
            neck_outputs = [neck_outputs]

        detections, detector_losses = self.box_head(neck_outputs, targets)

        return detections, x_lt, memory_long


class SSDXMemInference:
    def __init__(self,
                 weights: str,
                 cfg,
                 device,
                 score_th):
        self.dataset_root = DatasetCatalog.DATA_DIR
        self.anno_root = os.path.join(self.dataset_root, DatasetCatalog.DATASETS['dark_vid_val']['ann_file'])
        self.model = SSDXMemDetector(cfg)
        if len(weights):
            self.model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))['model'])

        self.model.to(device)
        self.model.eval()
        self.memory_buffer = []
        self.transforms = build_transforms(cfg, is_train=False)
        self.device = device
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        self.score_th = score_th
        self.class_names = DarkVIDDataset.classes
        self.class_maps = {v: k for k, v in enumerate(self.class_names)}

    def _forward_clip(self, video_clip: List[torch.Tensor]):
        tensor = torch.stack(video_clip)
        assert tensor.shape == (self.frame_len, 3, 320, 320)
        tensor = torch.unsqueeze(tensor, 0)

        if len(self.memory_buffer) == 0:
            result, x_lt, memory_long = self.model(tensor, None, None)
        else:
            memory = torch.stack(self.memory_buffer).transpose(0, 1)
            result, x_lt, memory_long = self.model(tensor, memory, None)
            x_lt = x_lt.squeeze(2)
            memory_long = memory_long.mean(2)

        return result, x_lt, memory_long

    def _result_postprocess(self, result, wh):
        w, h = wh
        result = result[0].resize((w, h)).to("cpu").numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > 0.5
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'classes: {}'.format(labels)
            ]
        )
        print(meters)
        print(scores)

        return boxes, labels, scores

    def evaluate(self, video):
        video_name = video.split(os.path.sep)[-1].split('.avi')[0]

        cap = cv.VideoCapture(video)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

        target_boxes, target_labels = [], []
        tensor_list = []
        results = []
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tensor = self.transforms(frame)[0]
            tensor = tensor.to(device=self.device, dtype=torch.float32)
            tensor_list.append(tensor)

            if len(tensor_list) == self.frame_len:
                with torch.no_grad():
                    result, x_lt, memory_long = self._forward_clip(tensor_list)

                if len(self.memory_buffer) == 0:
                    self.memory_buffer = [x_lt, x_lt, memory_long]
                else:
                    self.memory_buffer.pop(0)
                    self.memory_buffer.insert(1, x_lt)
                    self.memory_buffer.pop()
                    self.memory_buffer.append(memory_long)

                results.append(result[0])

                curr_target_boxes, curr_target_labels = self._parse_anno_file(video_name, frame_counter, (width, height))
                target_boxes.append(curr_target_boxes)
                target_labels.append(curr_target_labels)

                tensor_list.clear()

            frame_counter += 1

        targets = Container(
            boxes=target_boxes,
            labels=target_labels
        )

        return results, targets

    def _parse_anno_file(self, video_name, keyframe_idx, img_wh):
        df = pd.read_csv(os.path.join(self.anno_root, video_name + '.csv'), header=None)
        boxes, labels = [], []

        for row in df.index:
            label_info = df.loc[row]
            if int(keyframe_idx) == label_info[0]:
                curr_box = label_info[2:6]
                curr_label = label_info[10]

                boxes.append(np.array(curr_box).astype(np.float64))
                labels.append(int(self.class_maps[curr_label]))

        for i, box in enumerate(boxes):
            box[[0, 2]] = box[[0, 2]] / img_wh[0] * 320.0
            box[[1, 3]] = box[[1, 3]] / img_wh[1] * 320.0
            boxes[i][2] = box[0] + box[2]
            boxes[i][3] = box[1] + box[3]
        return boxes, labels

    def __call__(self, video, output_dir):
        cap = cv.VideoCapture(video)
        fps = cap.get(cv.CAP_PROP_FPS)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

        video_name = video.split(os.path.sep)[-1]

        writer = cv.VideoWriter(os.path.join(output_dir, video_name),
                                cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                float(fps),
                                (int(width), int(height)),
                                True)

        tensor_list = []
        boxes, labels, scores = [], [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tensor = self.transforms(frame)[0]
            tensor = tensor.to(device=self.device, dtype=torch.float32)
            tensor_list.append(tensor)

            if len(tensor_list) == self.frame_len:
                with torch.no_grad():
                    result, x_lt, memory_long = self._forward_clip(tensor_list)
                result = self._result_postprocess(result, (width, height))

                if len(self.memory_buffer) == 0:
                    self.memory_buffer = [x_lt, x_lt, memory_long]
                else:
                    self.memory_buffer.pop(0)
                    self.memory_buffer.insert(1, x_lt)
                    self.memory_buffer.pop()
                    self.memory_buffer.append(memory_long)

                boxes, labels, scores = result
                tensor_list.clear()

            drawn_image = draw_boxes(frame, boxes, labels,
                                     scores, self.class_names).astype(np.uint8)

            writer.write(drawn_image)


if __name__ == '__main__':
    config_path = './configs/resnet/dark/darknet_ssd320_dark_tma_pan.yaml'
    weight_path = "./outputs/5_classes_0818/model_038000.pth"

    cfg.merge_from_file(config_path)
    cfg.freeze()
    print("Loaded configuration file {}".format(config_path))
    with open(config_path, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    infer = SSDXMemInference(weight_path, cfg, device="cuda:0", score_th=0.3)

    video_path = "/mnt/h/Dataset/2.Carotid-Artery/2.Object-Detection/20230718/videos/test/202211161126240343VAS.avi"
    infer(video_path, "./demo_results/")

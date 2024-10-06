import math
from ssd.config import cfg
import numpy as np

import torch
import cv2 as cv
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F
from ssd.modeling.detector.ssd_detector import SSDLSTPANDetector
from ssd.data.transforms import build_transforms
from typing import List
from vizer.draw import draw_boxes
from ssd.data.datasets.dark_lt_vid import DarkVIDDataset
from ssd.config.path_catlog import DatasetCatalog
from ssd.structures.container import Container
from vis_cam import GradCAM


class SSDXMemDetector(SSDLSTPANDetector):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.grad_cam = GradCAM(self.backbone, self.backbone.darknet.dark5[-1], True)

    # def forward_cam(self, curr_video_clip, memory_buffer=None, targets=None):
    #     cam = self.grad_cam.calculate_cam(curr_video_clip)
    #     self.grad_cam.show_cam_on_image(curr_video_clip, cam)

    def forward(self, curr_video_clip, memory_buffer=None, targets=None):
        if memory_buffer is None:
            curr_feat = self.forward_backbone_attn(curr_video_clip)
            curr_xs = curr_feat["dark5"]
            _, curr_xs = self.forward_ltam(curr_xs.unsqueeze(0), torch.cat([curr_xs.unsqueeze(0) for _ in range(3)], dim=1))
            b, t, c, h, w = curr_xs.shape
            curr_xs = curr_feat["dark5"].view(b * t, c, h, w)
            # curr_xs = curr_xs.reshape(curr_xs.shape[0], curr_xs.shape[1], self.cfg.MODEL.PRIORS.FEATURE_MAPS[2], self.cfg.MODEL.PRIORS.FEATURE_MAPS[2])
            x_lt, memory_long = curr_xs, curr_xs
            curr_feat["dark5"] = curr_xs
            neck_outputs = self.backbone.forward_pan(curr_feat)
            # neck_outputs = [curr_feat["dark3"][-2:-1], curr_feat["dark4"][-2:：], curr_feat["dark5"]]
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
        # self.model = SSDXMemDetector(cfg)
        self.model = SSDLSTPANDetector(cfg)
        if len(weights):
            self.model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))['model'])

        self.memory_buffer = []
        self.transforms = build_transforms(cfg, is_train=False)
        self.device = device
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        self.score_th = score_th
        self.class_names = DarkVIDDataset.display_classes
        self.class_maps = {v: k for k, v in enumerate(self.class_names)}

        self.model.to(device)
        self.model.eval()
        self.feature_maps = dict()
        self.gradient = dict()
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps['value'] = output
            return None

        def backward_hook(module, grad_input, grad_output):
            self.gradient['value'] = grad_output[0]
            return None

        # target_layer_module = self.model.backbone.darknet.dark4[1].conv3.act
        target_layer_module = self.model.backbone.pan.C3_p3.conv3.conv
        # target_layer_module = self.model.backbone.darknet.dark5[3].conv3.act
        target_layer_module.register_forward_hook(forward_hook)
        target_layer_module.register_backward_hook(backward_hook)

    def _forward_detector(self, curr_video_clip, targets=None):
        if not len(self.memory_buffer):
            with torch.no_grad():
                curr_feat = self.model.forward_backbone_attn(curr_video_clip)
            curr_xs = curr_feat["dark5"]
            _, curr_xs = self.model.forward_ltam(curr_xs.unsqueeze(0), torch.cat([curr_xs.unsqueeze(0) for _ in range(3)], dim=1))
            # curr_xs = curr_xs.unsqueeze(1)
            b, t, c, h, w = curr_xs.shape
            curr_xs = curr_feat["dark5"].view(b * t, c, h, w)
            # curr_xs = curr_xs.reshape(curr_xs.shape[0], curr_xs.shape[1], self.cfg.MODEL.PRIORS.FEATURE_MAPS[2], self.cfg.MODEL.PRIORS.FEATURE_MAPS[2])
            x_lt, memory_long = curr_xs, curr_xs
            curr_feat["dark5"] = curr_xs
            neck_outputs = self.model.backbone.forward_pan(curr_feat)
            # neck_outputs = [curr_feat["dark3"][-2:-1], curr_feat["dark4"][-2:：], curr_feat["dark5"]]
        else:
            ndim = curr_video_clip.ndim
            assert ndim == 5, \
                "Expected dimension of previous images should be 5, got {}".format(
                    ndim)
            with torch.no_grad():
                curr_feat = self.model.forward_backbone_attn(curr_video_clip)
            curr_x0 = curr_feat['dark5']
            memory_long, x_lt = self.model.forward_ltam(curr_x0.contiguous().unsqueeze(0), torch.cat(self.memory_buffer, 1))
            if x_lt.ndim == 5:
                x_lt = x_lt.flatten(0, 1)
            curr_feat["dark5"] = x_lt
            neck_outputs = self.model.backbone.forward_pan(curr_feat)

            # return neck_outputs, x_lt, memory_long
            # neck_outputs, x_lt, memory_long = self.model.forward_infer(curr_video_clip, memory_buffer)

        if isinstance(neck_outputs, torch.Tensor):
            neck_outputs = [neck_outputs]

        detections, detector_losses = self.model.box_head(neck_outputs, targets)

        return detections, x_lt, memory_long

    def _forward_clip(self, video_clip: List[torch.Tensor]):
        tensor = torch.stack(video_clip).to(device=self.device)

        assert tensor.shape == (self.frame_len, 3, 320, 320)
        tensor = torch.unsqueeze(tensor, 0)

        if len(self.memory_buffer) == 0:
            result, x_lt, memory_long = self._forward_detector(tensor, None)
        else:
            if self.memory_buffer[0].ndim == 4:
                for i, mem in enumerate(self.memory_buffer):
                    self.memory_buffer[i] = mem.flatten(2).transpose(1, 2)

            # memory = torch.concat(self.memory_buffer, dim=1)
            result, x_lt, memory_long = self._forward_detector(tensor, None)
            x_lt = x_lt.squeeze(2)
            if memory_long.shape[1] == 300:
                b, n, c = memory_long.shape
                memory_long = memory_long.reshape(b, 3, 10, 10, c).mean(1).reshape(b, -1, c)

            elif memory_long.shape[1] == 100:
                b, n, c = memory_long.shape
                memory_long = memory_long.reshape(b, 1, 10, 10, c).mean(1).reshape(b, -1, c)

            else:
                if memory_long.ndim == 4:
                    memory_long = memory_long.unsqueeze(1)

        return result, x_lt.unsqueeze(1), memory_long

    def _result_postprocess(self, result, wh):
        w, h = wh
        result = result[0].resize((w, h))
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > self.score_th
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

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            tensor = self.transforms(frame)[0]
            tensor = tensor.to(device=self.device, dtype=torch.float32)
            tensor_list.append(tensor)

            if len(tensor_list) == self.frame_len:
                curr_target_boxes, curr_target_labels = self._parse_anno_file(video_name, frame_counter, (width, height))

                # if len(curr_target_labels):
                result, x_lt, memory_long = self._forward_clip(tensor_list)

                if len(self.memory_buffer) < 1:
                    if x_lt.ndim == 4:
                        x_lt = x_lt.unsqueeze(1)
                    self.memory_buffer = [x_lt, x_lt, x_lt]
                else:
                    self.memory_buffer.pop(0)
                    self.memory_buffer.insert(1, x_lt)
                    self.memory_buffer.pop()
                    self.memory_buffer.append(memory_long)

                if len(curr_target_labels):
                    target_boxes.append(curr_target_boxes)
                    target_labels.append(curr_target_labels)
                    results.append(result[0])
                # else:
                #     pass

                tensor_list.pop(0)

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
                if curr_label != "__background__":
                    boxes.append(np.array(curr_box).astype(np.float64))
                    labels.append(int(self.class_maps[curr_label]))

        if len(boxes):
            for i, box in enumerate(boxes):
                box[[0, 2]] = box[[0, 2]] / img_wh[0] * 320.0
                box[[1, 3]] = box[[1, 3]] / img_wh[1] * 320.0
                boxes[i][2] = box[0] + box[2]
                boxes[i][3] = box[1] + box[3]

        return boxes, labels

    @staticmethod
    def normalize_heatmap(blank_map):
        saliency_map_min, saliency_map_max = blank_map.min(), blank_map.max()
        blank_map = (blank_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
        blank_map = np.clip(blank_map * 255 + 0.5, 0, 255).astype(np.uint8)
        blank_map = cv.applyColorMap(blank_map, cv.COLORMAP_JET)
        blank_map = cv.cvtColor(blank_map, cv.COLOR_RGB2BGR)

        return blank_map

    @staticmethod
    def process_heatmap(blank_map, saliency_map):
        img_h, img_w = blank_map.shape
        saliency_map = F.upsample(saliency_map, size=(img_h, img_w), mode='bilinear', align_corners=False)
        saliency_map = saliency_map.detach().cpu().numpy()[0][0]
        blank_map = cv.add(blank_map, saliency_map)

        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
        saliency_map = np.clip(saliency_map * 255 + 0.5, 0, 255).astype(np.uint8)
        saliency_map = cv.applyColorMap(saliency_map, cv.COLORMAP_JET)
        saliency_map = cv.cvtColor(saliency_map, cv.COLOR_RGB2BGR)

        return blank_map, saliency_map

    def draw_heatmap_per_box(self, frame, target_boxes, saliency_maps):
        img_h, img_w, _ = frame.shape
        ratio_h, ratio_w = img_h / 320, img_w / 320
        target_boxes[:, 0::2] = target_boxes[:, 0::2] * ratio_w
        target_boxes[:, 1::2] = target_boxes[:, 1::2] * ratio_h
        # for target_box, saliency_map in zip(target_boxes, saliency_maps):

    def generate_heatmap(self, frame, boxes, labels, scores):
        mask = scores > 0.5
        target_scores = scores[mask]
        target_boxes = boxes[mask]
        target_labels = labels[mask]
        blank_saliency_map = np.zeros(frame.shape[:2], dtype=np.float32)
        saliency_maps = []

        for target_box, target_score, target_label in zip(target_boxes, target_scores, target_labels):
            if target_label == 5:
                self.model.zero_grad()
                target_score.backward(retain_graph=True)

                # gradients = self.gradient.mean(dim=(1, 2), keepdim=True)
                gradients = self.gradient['value']
                activations = self.feature_maps['value'][-1].unsqueeze(0)
                t, c, h, w = gradients.shape
                alpha = gradients.view(t, c, -1).mean(2)
                weights = alpha.view(t, c, 1, 1).mean(0).unsqueeze(0)
                saliency_map = (weights * activations).sum(1, keepdim=True)
                saliency_map = F.relu(saliency_map)
                blank_saliency_map, saliency_map = self.process_heatmap(blank_saliency_map, saliency_map)
                saliency_maps.append(saliency_map)

        blank_saliency_map = self.normalize_heatmap(blank_saliency_map)

        return frame, blank_saliency_map, saliency_maps

    def __call__(self, video, output_dir):
        cap = cv.VideoCapture(video)
        fps = cap.get(cv.CAP_PROP_FPS)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

        video_name = video.split(os.path.sep)[-1]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        writer = cv.VideoWriter(os.path.join(output_dir, video_name),
                                cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                float(fps),
                                (int(width), int(height)),
                                True)
        frame_counter = 0

        tensor_list = []
        boxes, labels, scores = [], [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            tensor = self.transforms(frame)[0]
            tensor_list.append(tensor)

            if len(tensor_list) == self.frame_len:
                self.model.zero_grad()
                result, x_lt, memory_long = self._forward_clip(tensor_list)

                if x_lt.ndim == 4:
                    x_lt = x_lt.unsqueeze(1)
                result = self._result_postprocess(result, (width, height))

                if len(self.memory_buffer) < 1:
                    self.memory_buffer = [x_lt, x_lt, x_lt]
                else:
                    # self.memory_buffer.pop(0)
                    # self.memory_buffer.append(x_lt)
                    prev_lt_feat = self.memory_buffer.pop(0)
                    self.memory_buffer.insert(1, x_lt)
                    prev_memory_feat = self.memory_buffer.pop()
                    self.memory_buffer.append(memory_long)

                    del prev_lt_feat, prev_memory_feat

                # if frame_counter % 30 * self.frame_len == 0 or len(self.memory_buffer) == 0:
                #     x_lt = x_lt.flatten(2).transpose(1, 2)
                #     self.memory_buffer = [x_lt]

                # if frame_counter * self.frame_len % 30 == 0 or len(self.memory_buffer) == 0:
                #     if len(self.memory_buffer) == 0:
                #         x_lt = x_lt.flatten(2).transpose(1, 2)
                #         self.memory_buffer = [x_lt]
                #     else:
                #         self.memory_buffer.append(x_lt)
                #         self.memory_buffer = [self.model.memory_decoder(torch.cat(self.memory_buffer, dim=2))]

                boxes, labels, scores = result
                res_img, heatmaps, saliency_maps = self.generate_heatmap(frame, boxes, labels, scores)
                res_img = cv.addWeighted(res_img, 0.7, heatmaps, 0.3, 0)
                plt.figure()
                plt.imshow(res_img)
                plt.savefig(os.path.join(output_dir, "gradcam_img_total_{:05d}.jpg".format(frame_counter)))

                tensor_list.clear()
                torch.cuda.empty_cache()
                if 'value' in self.feature_maps and 'value' in self.gradient:
                    del self.feature_maps['value'], self.gradient['value']

            frame_counter += 1

            # drawn_image = draw_boxes(frame, boxes, labels,
            #                          scores, tuple(self.class_names), width=6).astype(np.uint8)
            # drawn_image = cv.cvtColor(drawn_image, cv.COLOR_RGB2BGR)
            #
            # writer.write(drawn_image)


if __name__ == '__main__':
    config_path = './configs/resnet/dark/darknet_ssd320_mem_l.yaml'
    weight_path = "./outputs/240103_carotid_l/last_iteration_model.pth"

    cfg.merge_from_file(config_path)
    cfg.freeze()
    print("Loaded configuration file {}".format(config_path))
    with open(config_path, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    infer = SSDXMemInference(weight_path, cfg, device="cuda:0", score_th=0.25)

    # video_path = "/mnt/h/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20230825/videos/test/USm.1.2.410.200001.1.1185.2450099237.3.20230323.1140009474.561.11_Set AETitle.avi"
    video_path = "/media/scj/新加卷/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20240725/videos/external_test/EBDC771E-F2A8-4D49-AE6E-0397C4FCCFC5.avi"
    # video_path = "/media/scj/新加卷/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20240725/videos/valid/7441E58F-6BE3-4C0B-AC2F-6E29507999EC.avi"
    # video_path = "/media/scj/新加卷/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20240725/videos/external_test/N8NDH1O2.avi"
    if os.path.isdir(video_path):
        for video in os.listdir(video_path):
            curr_video_path = os.path.join(video_path, video)
            infer(curr_video_path, "./demo_results/exp_for_paper_plaque/gradcam")
    else:
        infer(video_path, "./demo_results/exp_for_paper_plaque/gradcam")

    # video_path = "/mnt/h/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20230825/videos/valid/202211161128190344VAS.avi"
    # video_path = ""

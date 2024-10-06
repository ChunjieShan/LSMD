import onnxruntime as ort
import numpy as np
import cv2 as cv
import os

from vizer.draw import draw_boxes
from ssd.data.datasets.dark_xmem_lst_vid import DarkVIDDataset
from ssd.data.transforms import build_transforms
from typing import List


class SSDXMemONNXInference:
    def __init__(
        self,
        cfg,
        device,
        score_th,
        backbone_onnx_model="../onnx_output/backbone.onnx",
        stam_onnx_model="../onnx_output/stam.onnx",
        ltam_onnx_model="../onnx_output/ltam.onnx",
        pan_onnx_model="../onnx_output/pan.onnx",
    ):
        self.memory_buffer = []
        self.transforms = build_transforms(cfg, is_train=False)
        self.device = device
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        self.score_th = score_th
        self.nms_th = 0.3
        self.class_names = DarkVIDDataset.classes

        (
            self.backbone_model,
            self.stam_model,
            self.ltam_model,
            self.pan_model,
        ) = self._init_all_onnx_models(
            backbone_onnx_model, stam_onnx_model, ltam_onnx_model, pan_onnx_model
        )

    @staticmethod
    def _init_onnx_engine(onnx_model_path, device):
        """
        Initialize ONNX session
        :param device: device you want to run at;
        :param onnx_model_path: str
        :return:
        """
        providers = None
        if device == "gpu":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 2GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                )
            ]
        elif device == "cpu" or None:
            providers = ["CPUExecutionProvider"]

        sess = ort.InferenceSession(onnx_model_path, providers=providers)
        return sess

    def _init_all_onnx_models(self, backbone_path, stam_path, ltam_path, pan_path):
        backbone_onnx_model = self._init_onnx_engine(backbone_path, device="gpu")
        stam_onnx_model = self._init_onnx_engine(stam_path, device="gpu")
        ltam_onnx_model = self._init_onnx_engine(ltam_path, device="gpu")
        pan_onnx_model = self._init_onnx_engine(pan_path, device="gpu")

        return backbone_onnx_model, stam_onnx_model, ltam_onnx_model, pan_onnx_model

    @staticmethod
    def _preprocess_image(image: np.ndarray):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (320, 320))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        mean = np.array([123.0, 117.0, 104.0], dtype=np.float32).reshape(1, -1, 1, 1)
        image = image.astype(np.float32)
        image = image - mean

        return image

    def _backbone_infer(self, input: np.array):
        """
        ONNX engine inference;
        :param session: onnxruntime inference session;
        :param img: preprocessed images;
        :return:
        """
        input_tensor = self.backbone_model.get_inputs()
        output_tensor = [node.name for node in self.backbone_model.get_outputs()]
        results = self.backbone_model.run(output_tensor, {input_tensor[0].name: input})

        return results

    def forward_backbone(self, image):
        input_tensor = self._preprocess_image(image)
        assert input_tensor.shape == (1, 3, 320, 320)

        result = self._backbone_infer(input_tensor)

        return result

    def _stam_infer(self, input: np.array):
        """
        ONNX engine inference;
        :param session: onnxruntime inference session;
        :param img: preprocessed images;
        :return:
        """
        input_tensor = self.stam_model.get_inputs()
        output_tensor = [node.name for node in self.stam_model.get_outputs()]
        results = self.stam_model.run(output_tensor, {input_tensor[0].name: input})

        return results

    def forward_stam(self, stam_feat):
        result = self._stam_infer(stam_feat)

        return result

    def _ltam_infer(self, ltam_q: np.ndarray, ltam_enc: np.ndarray):
        """
        ONNX engine inference;
        :param session: onnxruntime inference session;
        :param img: preprocessed images;
        :return:
        """
        input_tensor = self.ltam_model.get_inputs()
        output_tensor = [node.name for node in self.ltam_model.get_outputs()]
        results = self.ltam_model.run(
            output_tensor,
            {input_tensor[0].name: ltam_q, input_tensor[1].name: ltam_enc},
        )

        return results

    def forward_ltam(self, ltam_q, ltam_enc):
        result = self._ltam_infer(ltam_q, ltam_enc)

        return result

    def _pan_infer(self, x0: np.ndarray, x1: np.ndarray, x2: np.ndarray):
        """
        ONNX engine inference;
        :param session: onnxruntime inference session;
        :param img: preprocessed images;
        :return:
        """
        input_tensor = self.pan_model.get_inputs()
        output_tensor = [node.name for node in self.pan_model.get_outputs()]
        results = self.pan_model.run(
            output_tensor,
            {
                input_tensor[0].name: x0,
                input_tensor[1].name: x1,
                input_tensor[2].name: x2,
            },
        )

        return results

    def forward_pan(self, x0, x1, x2):
        result = self._pan_infer(x0, x1, x2)

        return result

    def _nms(self, boxes, scores, labels):
        indices = np.argsort(scores)[::-1]
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]

        keep_boxes = []
        keep_scores = []
        keep_labels = []

        if len(scores) <= 1:
            return boxes, scores, labels

        while len(scores) > 0:
            # 保留当前得分最高的框
            keep_boxes.append(boxes[0])
            keep_scores.append(scores[0])
            keep_labels.append(labels[0])

            # 计算当前框与其余框的重叠区域的左上角和右下角坐标
            overlap_left = np.maximum(boxes[0, 0], boxes[1:, 0])
            overlap_top = np.maximum(boxes[0, 1], boxes[1:, 1])
            overlap_right = np.minimum(boxes[0, 2], boxes[1:, 2])
            overlap_bottom = np.minimum(boxes[0, 3], boxes[1:, 3])

            # 计算重叠区域的宽度和高度
            overlap_width = np.maximum(0, overlap_right - overlap_left + 1)
            overlap_height = np.maximum(0, overlap_bottom - overlap_top + 1)

            # 计算重叠区域的面积
            overlap_area = overlap_width * overlap_height

            # 计算IoU（交并比）
            box_area = (boxes[0, 2] - boxes[0, 0] + 1) * (boxes[0, 3] - boxes[0, 1] + 1)
            iou = overlap_area / (
                box_area
                + (boxes[1:, 2] - boxes[1:, 0] + 1) * (boxes[1:, 3] - boxes[1:, 1] + 1)
                - overlap_area
            )

            # 保留IoU小于阈值的框
            indices = np.where(iou <= self.nms_th)[0]
            boxes = boxes[indices + 1]
            scores = scores[indices + 1]
            labels = labels[indices + 1]

        return np.array(keep_boxes), np.array(keep_scores), np.array(keep_labels)

    def _result_postprocess(self, boxes, scores, labels, wh):
        w, h = wh

        indices = scores > self.score_th
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        boxes, scores, labels = self._nms(boxes, scores, labels)

        boxes /= 320.0
        boxes[:, 0::2] = boxes[:, 0::2] * float(w)
        boxes[:, 1::2] = boxes[:, 1::2] * float(h)

        meters = " | ".join(
            ["objects {:02d}".format(len(boxes)), "classes: {}".format(labels)]
        )
        print(meters)
        print(scores)

        return boxes, scores, labels

    def __call__(self, video, output_dir):
        cap = cv.VideoCapture(video)
        fps = cap.get(cv.CAP_PROP_FPS)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

        video_name = video.split(os.path.sep)[-1]

        writer = cv.VideoWriter(
            os.path.join(output_dir, video_name),
            cv.VideoWriter_fourcc("X", "V", "I", "D"),
            float(fps),
            (int(width), int(height)),
            True,
        )

        tensor_list = []
        x0_list = []
        boxes, labels, scores = [], [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            backbone_output = self.forward_backbone(frame)

            x0_list.append(backbone_output[2])
            tensor_list.append(backbone_output)

            if len(tensor_list) == 3:
                stam_feat = np.stack(x0_list)
                stam_feat = np.transpose(stam_feat, (1, 2, 0, 3, 4))
                stam_q, stam_enc = self.forward_stam(stam_feat)
                stam_enc = np.mean(stam_enc, 2)
                stam_enc = np.expand_dims(stam_enc, 2)

                if not len(self.memory_buffer):
                    self.memory_buffer = [stam_q, stam_q, stam_enc]

                ltam_enc = np.concatenate(self.memory_buffer, axis=2)
                # ltam_enc = np.transpose(ltam_enc, (1, 2, 0, 3, 4))
                ltam_lt, mem = self.forward_ltam(ltam_q=stam_q, ltam_enc=ltam_enc)
                mem = np.mean(mem, 2)
                mem = np.expand_dims(mem, 2)

                self.memory_buffer.pop(1)
                self.memory_buffer.insert(0, ltam_lt)
                self.memory_buffer.pop()
                self.memory_buffer.append(mem)

                x0 = ltam_lt

                curr_xs = tensor_list[-1]
                curr_xs[2] = np.squeeze(stam_q, 2)
                curr_xs = curr_xs[::-1]

                detections = self.forward_pan(*curr_xs)
                boxes, scores, labels = detections

                boxes, scores, labels = self._result_postprocess(
                    boxes, scores, labels, (width, height)
                )

                tensor_list.clear()
                x0_list.clear()

            drawn_image = draw_boxes(
                frame, boxes, labels, scores, class_name_map=self.class_names
            )
            writer.write(drawn_image)


if __name__ == "__main__":
    from ssd.config import cfg

    config_path = "../configs/resnet/dark/darknet_ssd320_dark_tma_pan.yaml"
    cfg.merge_from_file(config_path)
    cfg.freeze()
    print("Loaded configuration file {}".format(config_path))
    with open(config_path, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    onnx_inferencer = SSDXMemONNXInference(cfg, device="gpu", score_th=0.3)
    video_path = "/mnt/d/20230718/videos/valid/USm.1.2.410.200001.1.1185.2450099237.3.20230316.1111130387.228.5_Set AETitle.avi"
    onnx_inferencer(video_path, "../demo_results/")

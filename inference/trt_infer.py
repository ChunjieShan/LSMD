import tensorrt as trt
import cv2 as cv
import numpy as np
import os
import pycuda.autoinit
import time
import pycuda.driver as cuda
from vizer.draw import draw_boxes

from ssd.data.datasets.dark_xmem_lst_vid import DarkVIDDataset
from typing import List


class TensorRTInference:
    def __init__(
        self,
        cfg,
        device,
        score_th,
        backbone_engine="./trt_engine/backbone_sim.engine",
        stam_engine="./trt_engine/stam_sim.engine",
        ltam_engine="./trt_engine/ltam_sim.engine",
        pan_engine="./trt_engine/pan_sim.engine",
    ):
        self.memory_buffer = []
        self.device = device
        self.frame_len = cfg.DATASETS.FRAME_LENGTH
        self.score_th = score_th
        self.nms_th = 0.3
        self.class_names = DarkVIDDataset.classes

        (
            self.backbone_engine,
            self.stam_engine,
            self.ltam_engine,
            self.pan_engine,
        ) = self._init_all_trt_engine(
            backbone_engine, stam_engine, ltam_engine, pan_engine
        )

    def _init_all_trt_engine(
        self, backbone_engine, stam_engine, ltam_engine, pan_engine
    ):
        backbone_trt_engine = self._init_engine(backbone_engine)
        print("[I] Backbone engine created")
        stam_trt_engine = self._init_engine(stam_engine)
        print("[I] STAM engine created")
        ltam_trt_engine = self._init_engine(ltam_engine)
        print("[I] LTAM engine created")
        pan_trt_engine = self._init_engine(pan_engine)
        print("[I] PAN and head engine created")

        return backbone_trt_engine, stam_trt_engine, ltam_trt_engine, pan_trt_engine

    @staticmethod
    def _init_engine(engine_path):
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())

        context = model.create_execution_context()

        return context

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

    def _backbone_infer(self, input: np.ndarray, input_shape, output_shapes: List[int]):
        """
        TensorRT engine inference;
        :param context: TensorRT context;
        :param preprocessed_img: images been preprocessed
        :return:
        """
        host_in = cuda.pagelocked_empty(input_shape, dtype=np.float32)
        np.copyto(host_in, input.ravel())
        host_out_40 = cuda.pagelocked_empty(output_shapes[0], dtype=np.float32)
        host_out_20 = cuda.pagelocked_empty(output_shapes[1], dtype=np.float32)
        host_out_10 = cuda.pagelocked_empty(output_shapes[2], dtype=np.float32)

        # engine = context.engine
        device_in = cuda.mem_alloc(host_in.nbytes)
        device_out_40 = cuda.mem_alloc(host_out_40.nbytes)
        device_out_20 = cuda.mem_alloc(host_out_20.nbytes)
        device_out_10 = cuda.mem_alloc(host_out_10.nbytes)

        bindings = [
            int(device_in),
            int(device_out_40),
            int(device_out_20),
            int(device_out_10),
        ]
        stream = cuda.Stream()

        start = time.time()
        # for _ in range(100):
        cuda.memcpy_htod_async(device_in, host_in, stream)
        self.backbone_engine.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(host_out_40, device_out_40, stream)
        cuda.memcpy_dtoh_async(host_out_20, device_out_20, stream)
        cuda.memcpy_dtoh_async(host_out_10, device_out_10, stream)
        stream.synchronize()
        print("Backbone infer:\t{} ms".format((time.time() - start) * 1000))
        return host_out_40, host_out_20, host_out_10

    def forward_backbone(self, image):
        input_tensor = self._preprocess_image(image)
        assert input_tensor.shape == (1, 3, 320, 320)

        result = self._backbone_infer(
            input_tensor,
            input_shape=1 * 3 * 320 * 320,
            output_shapes=[1 * 256 * 40 * 40, 1 * 512 * 20 * 20, 1 * 1024 * 10 * 10],
        )

        backbone_outputs = [
            np.reshape(result[0], (1, 256, 40, 40)),
            np.reshape(result[1], (1, 512, 20, 20)),
            np.reshape(result[2], (1, 1024, 10, 10)),
        ]

        return backbone_outputs

    def _stam_infer(self, input: np.ndarray, input_shape, output_shapes: List[int]):
        """
        TensorRT engine inference;
        :param context: TensorRT context;
        :param preprocessed_img: images been preprocessed
        :return:
        """
        host_in = cuda.pagelocked_empty(input_shape, dtype=np.float32)
        np.copyto(host_in, input.ravel())
        host_out_stam_q = cuda.pagelocked_empty(output_shapes[0], dtype=np.float32)
        host_out_stam_enc = cuda.pagelocked_empty(output_shapes[1], dtype=np.float32)

        # engine = context.engine
        device_in = cuda.mem_alloc(host_in.nbytes)
        device_out_stam_q = cuda.mem_alloc(host_out_stam_q.nbytes)
        device_out_stam_enc = cuda.mem_alloc(host_out_stam_enc.nbytes)

        bindings = [
            int(device_in),
            int(device_out_stam_q),
            int(device_out_stam_enc),
        ]
        stream = cuda.Stream()

        start = time.time()
        # for _ in range(100):
        cuda.memcpy_htod_async(device_in, host_in, stream)
        self.stam_engine.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(host_out_stam_q, device_out_stam_q, stream)
        cuda.memcpy_dtoh_async(host_out_stam_q, device_out_stam_enc, stream)
        stream.synchronize()
        print("STAM infer:\t{} ms".format((time.time() - start) * 1000))
        return host_out_stam_q, host_out_stam_enc

    def forward_stam(self, feat):
        assert feat.shape == (1, 1024, 3, 10, 10)

        result = self._stam_infer(
            feat,
            input_shape=1 * 1024 * 3 * 10 * 10,
            output_shapes=[1 * 1024 * 1 * 10 * 10, 1 * 1024 * 3 * 10 * 10],
        )

        stam_outputs = [
            np.reshape(result[0], (1, 1024, 1, 10, 10)),
            np.reshape(result[1], (1, 1024, 3, 10, 10)),
        ]

        return stam_outputs

    def _ltam_infer(
        self,
        inputs: List[np.ndarray],
        input_shapes: List[int],
        output_shapes: List[int],
    ):
        """
        TensorRT engine inference;
        :param context: TensorRT context;
        :param preprocessed_img: images been preprocessed
        :return:
        """
        host_in_ltam_q = cuda.pagelocked_empty(input_shapes[0], dtype=np.float32)
        host_in_ltam_enc = cuda.pagelocked_empty(input_shapes[1], dtype=np.float32)
        np.copyto(host_in_ltam_q, inputs[0].ravel())
        np.copyto(host_in_ltam_enc, inputs[1].ravel())

        host_out_ltam_lt = cuda.pagelocked_empty(output_shapes[0], dtype=np.float32)
        host_out_ltam_mem = cuda.pagelocked_empty(output_shapes[1], dtype=np.float32)

        # engine = context.engine
        device_in_ltam_q = cuda.mem_alloc(host_in_ltam_q.nbytes)
        device_in_ltam_enc = cuda.mem_alloc(host_in_ltam_enc.nbytes)
        device_out_ltam_lt = cuda.mem_alloc(host_out_ltam_lt.nbytes)
        device_out_ltam_mem = cuda.mem_alloc(host_out_ltam_mem.nbytes)

        bindings = [
            int(device_in_ltam_q),
            int(device_in_ltam_enc),
            int(device_out_ltam_lt),
            int(device_out_ltam_mem),
        ]
        stream = cuda.Stream()

        start = time.time()
        # for _ in range(100):
        cuda.memcpy_htod_async(device_in_ltam_q, host_in_ltam_q, stream)
        cuda.memcpy_htod_async(device_in_ltam_enc, host_in_ltam_enc, stream)
        self.ltam_engine.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(host_out_ltam_lt, device_out_ltam_lt, stream)
        cuda.memcpy_dtoh_async(host_out_ltam_mem, device_out_ltam_mem, stream)
        stream.synchronize()
        print("LTAM infer:\t{} ms".format((time.time() - start) * 1000))
        return host_out_ltam_lt, host_out_ltam_mem

    def forward_ltam(self, ltam_q, ltam_enc):
        assert ltam_q.shape == (1, 1024, 1, 10, 10)
        assert ltam_enc.shape == (1, 1024, 3, 10, 10)

        result = self._ltam_infer(
            [ltam_q, ltam_enc],
            input_shapes=[1 * 1024 * 1 * 10 * 10, 1 * 1024 * 3 * 10 * 10],
            output_shapes=[1 * 1024 * 1 * 10 * 10, 1 * 1024 * 3 * 10 * 10],
        )

        ltam_outputs = [
            np.reshape(result[0], (1, 1024, 1, 10, 10)),
            np.reshape(result[1], (1, 1024, 3, 10, 10)),
        ]

        return ltam_outputs

    def _pan_infer(
        self,
        inputs: List[np.ndarray],
        input_shapes: List[int],
        output_shapes: List[int],
    ):
        """
        TensorRT engine inference;
        :param context: TensorRT context;
        :param preprocessed_img: images been preprocessed
        :return:
        """
        host_in_feat_x0 = cuda.pagelocked_empty(input_shapes[0], dtype=np.float32)
        host_in_feat_x1 = cuda.pagelocked_empty(input_shapes[1], dtype=np.float32)
        host_in_feat_x2 = cuda.pagelocked_empty(input_shapes[2], dtype=np.float32)

        np.copyto(host_in_feat_x0, inputs[0].ravel())
        np.copyto(host_in_feat_x1, inputs[1].ravel())
        np.copyto(host_in_feat_x2, inputs[2].ravel())

        host_out_scores = cuda.pagelocked_empty(output_shapes[0], dtype=np.float32)
        host_out_labels = cuda.pagelocked_empty(output_shapes[1], dtype=np.int32)
        host_out_boxes = cuda.pagelocked_empty(output_shapes[2], dtype=np.float32)

        # engine = context.engine
        device_in_x0 = cuda.mem_alloc(host_in_feat_x0.nbytes)
        device_in_x1 = cuda.mem_alloc(host_in_feat_x1.nbytes)
        device_in_x2 = cuda.mem_alloc(host_in_feat_x2.nbytes)

        device_out_scores = cuda.mem_alloc(host_out_scores.nbytes)
        device_out_labels = cuda.mem_alloc(host_out_labels.nbytes)
        device_out_boxes = cuda.mem_alloc(host_out_boxes.nbytes)

        bindings = [
            int(device_in_x0),
            int(device_in_x1),
            int(device_in_x2),
            int(device_out_scores),
            int(device_out_labels),
            int(device_out_boxes),
        ]
        stream = cuda.Stream()

        start = time.time()
        # for _ in range(100):
        cuda.memcpy_htod_async(device_in_x0, host_in_feat_x0, stream)
        cuda.memcpy_htod_async(device_in_x1, host_in_feat_x1, stream)
        cuda.memcpy_htod_async(device_in_x2, host_in_feat_x2, stream)

        self.pan_engine.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(host_out_scores, device_out_scores, stream)
        cuda.memcpy_dtoh_async(host_out_labels, device_out_labels, stream)
        cuda.memcpy_dtoh_async(host_out_boxes, device_out_boxes, stream)
        stream.synchronize()
        print("PAN infer:\t{} ms".format((time.time() - start) * 1000))
        return host_out_scores, host_out_labels, host_out_boxes

    def forward_pan(self, x0, x1, x2):
        assert x0.shape == (1, 1024, 10, 10)
        assert x1.shape == (1, 512, 20, 20)
        assert x2.shape == (1, 256, 40, 40)

        result = self._pan_infer(
            [x0, x1, x2],
            input_shapes=[1 * 1024 * 10 * 10, 1 * 512 * 20 * 20, 1 * 256 * 40 * 40],
            output_shapes=[12600 * 5, 12600 * 5, 12600 * 5 * 4],
        )

        head_outputs = [result[0], result[1], np.reshape(result[2], (12600 * 5, 4))]

        return head_outputs

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
                scores, labels, boxes = detections

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

    config_path = "./configs/resnet/dark/darknet_ssd320_dark_tma_pan.yaml"
    cfg.merge_from_file(config_path)
    cfg.freeze()
    print("Loaded configuration file {}".format(config_path))
    with open(config_path, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    trt_inferencer = TensorRTInference(cfg, device="gpu", score_th=0.3)
    video_path = "/mnt/d/20230718/videos/valid/USm.1.2.410.200001.1.1185.2450099237.3.20230316.1111130387.228.5_Set AETitle.avi"
    trt_inferencer(video_path, "./demo_results/")

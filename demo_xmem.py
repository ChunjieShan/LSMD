import glob
import os
import time

import cv2 as cv
import torch
import torch.nn as nn
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset, COCOVIDDataset, DarkVIDDataset
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, video_dir, videos_dir=None, output_dir=None, dataset_type=None):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    elif dataset_type == "coco_vid":
        class_names = COCOVIDDataset.class_names
    elif dataset_type == "dark_vid":
        class_names = DarkVIDDataset.classes
    else:
        raise NotImplementedError('Not implemented now.')

    # device = torch.device(cfg.MODEL.DEVICE)
    device = torch.device("cuda:0")

    model = build_detection_model(cfg)
    model = model.to(device)
    # for name, param in model.named_parameters():
    #     print(param)
    #     break
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    # for name, param in model.named_parameters():
    #     print(param)
    #     break

    # image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    predict_single_video(model,
                         frame_len=3,
                         transforms=transforms,
                         video_path=video_dir,
                         class_names=class_names,
                         dest_video_path="./demo_results/",
                         score_threshold=score_threshold,
                         device=device)


def predict_single_video(model: nn.Module,
                         transforms,
                         device: torch.device,
                         video_path: str,
                         frame_len: int,
                         class_names,
                         score_threshold,
                         dest_video_path):
    cpu_device = torch.device("cpu")

    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    video_name = video_path.split(os.path.sep)[-1]

    writer = cv.VideoWriter(os.path.join(dest_video_path, video_name),
                            cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                            float(fps),
                            (int(width), int(height)),
                            True)

    tensor_list = []
    prev_tensor_list = []
    drawn_flag = False
    first_flag = True
    boxes, labels, scores = [], [], []
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = transforms(frame)[0]
        tensor = tensor.to(device=device, dtype=torch.float32)
        tensor_list.append(tensor)

        if len(tensor_list) == frame_len:
            tensor = torch.stack(tensor_list)
            assert tensor.shape == (frame_len, 3, 320, 320)
            tensor = torch.unsqueeze(tensor, 0)

            if first_flag:
                # feat, result = model.forward_curr_clips(curr_images=tensor, prev_feat=None)
                feat, result = model(tensor, tensor, None)
                first_flag = False
            else:
                prev_tensor = torch.stack(prev_tensor_list)
                prev_tensor = torch.unsqueeze(prev_tensor, 0)

                feat, result = model(prev_tensor, tensor, None)

            result = result[0].resize((width, height)).to(cpu_device).numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            indices = scores > score_threshold
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

            if counter % 2 == 0:
                prev_tensor_list = tensor_list
            tensor_list.clear()

            counter += 1
        drawn_image = draw_boxes(frame, boxes, labels,
                                 scores, class_names).astype(np.uint8)
        writer.write(drawn_image)


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        # default='./configs/resnet50_ssd320_coco_vid_middle_head.yaml',
        default='./configs/resnet/dark/darknet_ssd320_dark_tma_pan.yaml',
        # default="./configs/mobilenet_v3_ssd320_coco_vid_less_head.yaml",
        # default="./configs/mobilenet_v3_ssd320_coco_ours.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt",
                        type=str,
                        default='./outputs/darknet_ssd320_dark_vid_carotid_pan_0307/model_040000.pth',
                        # default='./outputs/resnet_ssd320_coco_vid_carotid_5_heads/model_015000.pth',
                        help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.4)
    # parser.add_argument("--images_dir",
    #                     default='F:/BME_New/Echocardiography_Datas/Quality_Assessment_object_detection/Dataset_of_ssd_v1/train/',
    #                     type=str, help='Specify an image dir to do prediction.')
    parser.add_argument("--images_dir",
                        # default="/home/scj/Projects/Code/Data/PSAX-A/single_frame_dataset/Dataset_ssd_v1/test/USM.1.2.840.113619.2.391.3780.1637312451.97.1.512_AURORA-003780.DCM",
                        type=str, help='Specify a image dir to do prediction.')

    parser.add_argument("--videos-dir",
                        # default="/home/scj/Projects/Code/Data/PSAX-MV/single_frame_dataset/Dataset_ssd_v2/test/",
                        # default="/home/scj/Projects/Code/Data/PSAX-A/single_frame_dataset/image_data/",
                        default="/home/scj/Code/Data/Carotid-Artery/data/OD-Labels/Dataset_ssd_v1/train",
                        type=str,
                        help='Specify the videos root directory.')

    parser.add_argument("--output_dir", default='./demo_results',
                        type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="dark_vid", type=str,
                        help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             # video_dir="/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/videos_1209/F666978B-9BF5-4E50-AC22-CD56AAD93A6F.avi",
             video_dir="/home/scj/Code/Data/2.Carotid-Artery/demo/USm.1.2.410.200001.1.1185.2450099237.3.20230215.1144512159.332.5_Set AETitle.avi",
             # video_dir="/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/20230201/20230201_cross/videos/0C0A5615-DE3B-4741-9210-3BA7157C646D.avi",
             videos_dir=None,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()

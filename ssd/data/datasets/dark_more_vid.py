import os
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import copy
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset
from PIL import Image
from ssd.structures.container import Container


def plot_img_for_debug(image, boxes, labels):
    img = copy.deepcopy(image)
    # img = np.transpose(img, (1, 2, 0))
    # w, h = img.size
    plt.figure()
    # img = cv.resize(img, (128, 128))
    # img.resize((320, 320), refcheck=False)
    plt.imshow(img)
    plt.title([label for label in labels])
    for box, label in zip(boxes, labels):
        if len(box) == 4:
            x1, y1, x2, y2 = box
        else:
            x1, y1, x2, y2, label = box
        # x1 = x1 * 320
        # y1 = y1 * 320
        # x2 = x2 * 320
        # y2 = y2 * 320
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), (x2 - x1), (y2 - y1), fill=False,
                          edgecolor='r', linewidth=3))

    plt.show()


class DarkVIDDataset(Dataset):
    classes = ['__background__', 'cross']

    def __init__(self, data_dir, label_dir, frame_len=None, transform=None, target_transform=None, remove_empty=False):
        self.classes = ['__background__', 'cross']
        self.num_classes = len(self.classes)
        self.class_map = {v: k for k, v in enumerate(self.classes)}

        super(DarkVIDDataset, self).__init__()
        self.frame_len = frame_len
        self.video_dir = data_dir
        self.label_dir = label_dir

        self.video_file_list = os.listdir(data_dir)
        self.video_file_list.sort()
        self.label_file_list = os.listdir(label_dir)
        self.label_file_list.sort()
        self.training_file_list, self.training_label_list = self._allocate_samples()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        curr_video_clip = self.training_file_list[index]
        curr_gt = self.training_label_list[index]
        keyframe_idx = int(curr_gt.split(os.path.sep)[-1].split(".")[0].split("_")[-1])
        curr_video_name = curr_video_clip[0].split(os.path.sep)[0]

        label_file = os.path.join(self.label_dir, curr_video_name + ".csv")
        img_list = self.read_videos(video_clip=curr_video_clip)
        boxes, labels = self.get_annotations(
            label_file, keyframe_idx, img_size=img_list[0].size[0:2])

        original_boxes = np.copy(boxes)
        # original_images = np.copy(np.array(img_list[-1]))

        tensor_img_list = []
        if self.transform:
            for idx, img in enumerate(img_list):
                # img = Image.fromarray(img)
                img = np.array(img)
                if idx == len(img_list) - 1:
                    img, boxes, labels = self.transform(img, original_boxes, labels)
                else:
                    img, _, _ = self.transform(img, boxes, labels)
                tensor_img_list.append(img)

        # plot_img_for_debug(original_images, original_boxes, labels)
        images = torch.stack(tensor_img_list)

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        targets = Container(
            boxes=boxes,
            labels=labels,
        )

        return images, targets, 0

    def __len__(self):
        return len(self.training_label_list)

    # def _allocate_samples(self):
    #     img_names_list = []
    #     gt_list = []
    #
    #     for image_dir_name in self.video_file_list:
    #         image_dir = os.path.join(self.video_dir, image_dir_name)
    #         image_names = os.listdir(image_dir)
    #         image_names.sort()
    #
    #         for idx in range(len(image_names) - self.frame_len):
    #             image_sample_names = image_names[idx:(idx + self.frame_len)]
    #             for curr_idx, image_sample_name in enumerate(image_sample_names):
    #                 image_sample_names[curr_idx] = os.path.join(image_dir_name, image_sample_name)
    #
    #             img_names_list.append(image_sample_names)
    #             gt_list.append(image_sample_names[-1])
    #
    #     return img_names_list, gt_list

    def _allocate_samples(self):
        img_names_list = []
        gt_list = []

        for image_dir_name in self.video_file_list:
            image_dir = os.path.join(self.video_dir, image_dir_name)
            df_label = pd.read_csv(os.path.join(self.label_dir, image_dir_name + ".csv"), header=None)
            keyframe_idxes_list = np.array(df_label.iloc[:, 0]).tolist()
            keyframe_idxes_list.sort()
            keyframe_idxes_list = list(set(keyframe_idxes_list))

            image_names = os.listdir(image_dir)
            image_names.sort()

            for idx in range(len(image_names) - self.frame_len):
                image_sample_names = image_names[idx:(idx + self.frame_len)]
                key_idx = int(image_sample_names[-1].split(".")[0].split("_")[-1])
                if key_idx in keyframe_idxes_list:
                    for curr_idx, image_sample_name in enumerate(image_sample_names):
                        image_sample_names[curr_idx] = os.path.join(image_dir_name, image_sample_name)

                    img_names_list.append(image_sample_names)
                    gt_list.append(image_sample_names[-1])

        return img_names_list, gt_list

    def get_annotations(self, label_file, keyframe_idx, img_size):
        df = pd.read_csv(label_file, header=None)
        boxes = []
        labels = []

        for row in df.index:
            label_info = df.loc[row]
            if int(keyframe_idx) == label_info[0]:
                curr_box = label_info[2:6]
                # curr_box = self._xywh2xyxy(curr_box)
                # curr_box.append(self.class_map[label_info[10]])

                boxes.append(np.array(curr_box).astype(np.float64))
                labels.append(self.class_map[label_info[10]])

        boxes = self.preprocess_box(np.array(boxes), img_size)

        return boxes, np.array(labels)

    def preprocess_box(self, boxes, img_shape):
        boxes = self._xywh2xyxy(boxes)
        # boxes = self._convert_coord_to_percent(boxes, img_shape)

        return boxes

    @staticmethod
    def _xywh2xyxy(boxes: np.array):
        boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

        return boxes

    @staticmethod
    def _convert_coord_to_percent(boxes: np.array, img_shape):
        img_w, img_h = img_shape
        boxes[:, [0, 2]] /= img_w
        boxes[:, [1, 3]] /= img_h

        return boxes

    def read_videos(self, video_clip):
        pil_imgs_list = []

        for img_file in video_clip:
            pil_img = Image.open(os.path.join(
                self.video_dir, img_file)).convert('RGB')
            pil_imgs_list.append(pil_img)

        return pil_imgs_list


if __name__ == '__main__':
    from ssd.config import cfg
    from ssd.data.build import build_transforms, build_target_transform

    cfg.merge_from_file(
        "../../../configs/resnet/dark/resnet50_ssd320_dark_vid_large_head.yaml")
    cfg.freeze()

    transform = build_transforms(cfg, True)
    target_transform = build_target_transform(cfg)

    dataset = DarkVIDDataset(data_dir="/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/20230201/20230201_cross/images",
                             label_dir="/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/20230201/20230201_cross/labels",
                             frame_len=3,
                             transform=transform,
                             target_transform=target_transform)
    for i, data in enumerate(dataset):
        print(i)
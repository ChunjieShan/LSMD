import copy
import os
import torch.utils.data
import numpy as np
import pandas as pd
import random
from PIL import Image
from typing import List, Dict
from ssd.data.transforms.transforms import ToPercentCoords
import cv2 as cv
import matplotlib.pyplot as plt

from ssd.structures.container import Container


def add_mask_on_specific_space(imgs: List[np.array],
                               boxes,
                               labels,
                               keyframe_idx,
                               frame_len: int = 32,
                               mask_class=None):
    if mask_class is None:
        random_class_idx = random.randint(0, len(labels) - 1)
        mask_class = [labels[random_class_idx]]

    class_idx = mask_class[0] - 1

    # original_img = np.copy(img)
    box = boxes[class_idx]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    key_img = imgs[keyframe_idx]
    # temp_img = np.copy(img)
    img_h, img_w, _ = key_img.shape

    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    w, h = x2 - x1, y2 - y1

    noise_block = np.random.normal(0, 0.01, (h, w // 2, 3)) * 255
    # temp_img[y1:(y1 + h), (x1 + w // 2):(x1 + w // 2 + w // 2), :] = noise_block[:, :, :]

    for i, img in enumerate(imgs):
        # origin_img = np.copy(img)
        if i % 2 == 0:
            img[y1:(y1 + h), (x1 + w // 2):(x1 + w // 2 + w // 2), :] = noise_block[:, :, :]
            imgs[i] = img

        # ============ for testing =============
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(origin_img)
        # plt.title("Original")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(img)
        # plt.title("Augmented({})".format(labels[class_idx]))
        # plt.gca().add_patch(
        #     plt.Rectangle((x1, y1), w, h, fill=False,
        #                   edgecolor='r', linewidth=3))
        #
        # plt.show()

    # imgs[keyframe_idx] = temp_img

    return imgs


class COCOVIDDataset(torch.utils.data.Dataset):
    # class_names = ('__background__', 'cross', 'vertical')
    # class_names = ('__background__', 'muscle', "LV", "MV", "RV", "LVF", "LVS", "LVB", "LVL")
    class_names = ('__background__', 'heart', "LV", "MV", "RV", "LVF", "LVS", "LVB", "LVL")

    def __init__(self, data_dir, ann_file, csv_file, frame_len=32, transform=None, target_transform=None,
                 remove_empty=False):
        from pycocotools.coco import COCO
        self.ann_file = ann_file
        self.coco = COCO(ann_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.remove_empty = remove_empty
        self.frame_len = frame_len
        self.convert_boxes_to_coords = ToPercentCoords()

        if self.remove_empty:
            # when training, images without annotations are removed.
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            # when testing, all images used.
            self.ids = list(self.coco.imgs.keys())
        coco_categories = sorted(self.coco.getCatIds())
        self.csv_dir = csv_file
        self.csv_files = os.listdir(csv_file)
        self.csv_frame_dict = self._get_frame_idx()
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        img_list, keyframe_idx = self._read_video(image_id)
        # if random.randint(0, 1):
        #     add_mask_on_specific_space(img_list, boxes, labels, keyframe_idx)

        # ============== For debugging ==============
        # img = copy.deepcopy(img_list[0])
        # print(image_id)
        # plt.figure()
        # # img = cv.resize(img, (300, 300))
        # # img.resize((320, 320), refcheck=False)
        # plt.imshow(img)
        # plt.title([label for label in labels])
        # for box, label in zip(boxes, labels):
        #     x1, y1, x2, y2 = box
        #     plt.gca().add_patch(
        #         plt.Rectangle((x1, y1), (x2 - x1), (y2 - y1), fill=False,
        #                       edgecolor='r', linewidth=3))
        #
        # plt.show()

        tensor_img_list = []
        if self.transform:
            for img in img_list:
                img, _, _ = self.transform(img, boxes, labels)
                tensor_img_list.append(img)

        _, boxes, labels = self.convert_boxes_to_coords(img_list[0], boxes, labels)

        tensor_img = torch.stack(tensor_img_list)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return tensor_img, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _get_frame_idx(self):
        csv_frame_dict = {}
        for csv_file in self.csv_files:
            label = pd.read_csv(os.path.join(self.csv_dir, csv_file), header=None)
            frame_idx = label.values[0, 0]
            csv_frame_dict[csv_file.split(".csv")[0]] = frame_idx

        return csv_frame_dict

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def get_img_info(self, index):
        image_id = self.ids[index]
        img_data = self.coco.imgs[image_id]
        return img_data

    def _read_video(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        # video_path = os.path.join(self.data_dir, file_name)
        # image = Image.open(video_path).convert("RGB")
        # image = np.array(image)
        return self._read_image(file_name)

    def _read_image(self, video_file_name):
        img_file_list = os.listdir(os.path.join(self.data_dir, video_file_name))
        # os.listdir() will return unsorted filenames list
        # So it is necessary to make the list in order
        img_file_list.sort()
        frame_idx = self.csv_frame_dict[video_file_name]
        total_frames = len(img_file_list)

        assert frame_idx <= total_frames, "[E] Frame index is larger than total frames!"

        stride = int(total_frames / self.frame_len)
        start_pos = frame_idx % stride
        start_pos = max(0, start_pos - 1)
        end_pos = start_pos + (self.frame_len * stride)

        if end_pos > total_frames:
            end_pos = total_frames + 1
        start_dist = (frame_idx - start_pos) // stride

        # dist_to_end = total_frames - frame_idx
        # if frame_idx >= self.frame_len:
        #     # dist_to_end = self.frame_len - (total_frames - frame_idx)
        #     if (dist_to_end >= 0) and (dist_to_end < (self.frame_len - 1)):
        #         start_dist = frame_idx - (total_frames - self.frame_len)
        #         # start_dist = random.randint((frame_idx - (total_frames - self.frame_len)), (total_frames - self.frame_len))
        #         assert start_dist >= 0, "[E] Start distance is smaller than 0!"
        #     else:
        #         start_dist = random.randint(0, self.frame_len - 1)
        #
        # else:
        #     if (dist_to_end >= 0) and (dist_to_end < (self.frame_len - 1)):
        #         start_dist = frame_idx - (total_frames - self.frame_len)
        #         # start_dist = random.randint((frame_idx - (total_frames - self.frame_len)), (total_frames - self.frame_len))
        #         assert start_dist >= 0, "[E] Start distance is smaller than 0!"
        #     else:
        #         start_dist = random.randint(0, frame_idx)
        #
        # start_pos = frame_idx - start_dist

        img_sample_list = img_file_list[start_pos:end_pos:stride]
        # img_sample_list = img_file_list[start_pos:(start_pos + self.frame_len)]
        # print(len(img_sample_list))
        assert len(img_sample_list) == self.frame_len, \
            f"[E] {video_file_name} got wrong! Expected sampling {self.frame_len} frames, got {len(img_sample_list)}"
        # assert frame_idx <= total_frames, "[E] Frame index is larger than total frames!"
        #
        # stride = total_frames // self.frame_len
        # start_pos = 0
        # if stride * self.frame_len < frame_idx:
        #     dist = frame_idx - stride * self.frame_len
        #     start_pos += dist
        #
        # end_pos = start_pos + self.frame_len * stride
        #
        # img_sample_list = img_file_list[start_pos:end_pos:stride]
        np_img_list = []
        for img_file in img_sample_list:
            pil_img = Image.open(os.path.join(self.data_dir, video_file_name, img_file)).convert("RGB")
            np_img = np.array(pil_img)
            np_img_list.append(np_img)

        return np_img_list, start_dist


if __name__ == '__main__':
    from ssd.config import cfg
    from ssd.data.build import build_transforms, build_target_transform

    cfg.merge_from_file(
        "/home/scj/Projects/Code/0.Frameworks/3.Video-Understanding/2.VID/22-10-24-SSD-3D-New-VID/configs/mobilenet_v3_ssd320_coco_ours.yaml")
    cfg.freeze()

    transform = build_transforms(cfg, True)
    target_transform = build_target_transform(cfg)

    dataset = COCOVIDDataset(data_dir="/home/scj/Projects/Code/Data/PSAX-A/single_frame_dataset/Dataset_ssd_v2/train",
                             ann_file="/home/scj/Projects/Code/Data/PSAX-A/single_frame_dataset/Dataset_ssd_v2/annotations/instances_train.json",
                             csv_file="/home/scj/Projects/Code/Data/PSAX-A/single_frame_dataset/labels",
                             transform=transform,
                             target_transform=target_transform)
    for i, data in enumerate(dataset):
        pass

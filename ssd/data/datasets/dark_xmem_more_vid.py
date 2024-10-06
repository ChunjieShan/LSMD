import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ssd.data.transforms import build_prev_clip_transform
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
    # classes = ['__background__', 'AO', 'HEART', 'IVS', 'LA', 'LV', 'MV', 'RA', 'RV', 'TV']
    classes = ['__background__', 'plaque']

    def __init__(self, data_dir, label_dir, frame_len=None, transform=None, prev_transform=None, target_transform=None,
                 remove_empty=False, is_random=False, is_overlap=False, sequence_num=2):
        self.classes = ['__background__', 'plaque']
        # self.classes = ['__background__', 'A5C_AO', 'A5C_HEART', 'A5C_IVS', 'A5C_LA', 'A5C_LV', 'A5C_MV', 'A5C_RA', 'A5C_RV', 'A5C_TV']
        self.num_classes = len(self.classes)
        self.class_map = {v: k for k, v in enumerate(self.classes)}

        super(DarkVIDDataset, self).__init__()
        self.sequence_num = sequence_num
        self.is_random = is_random
        self.frame_len = frame_len
        self.video_dir = data_dir
        self.label_dir = label_dir

        self.video_file_list = os.listdir(data_dir)
        self.video_file_list.sort()
        self.label_file_list = os.listdir(label_dir)
        self.label_file_list.sort()
        if is_overlap:
            self.training_file_list, self.training_label_list = self._allocate_samples()
        else:
            self.training_file_list, self.training_label_list = self.allocate_consecutive_samples()

        # self.training_file_list, self.training_label_list = self.allocate_consecutive_samples()
        self.remove_empty = remove_empty

        if prev_transform is None:
            raise AttributeError("[E] Previous video clip transform is None.")

        self.transform = transform
        self.prev_transform = prev_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data_clip = self.training_file_list[index]
        prev_videos_clips, curr_video_clip = data_clip
        curr_gt = self.training_label_list[index]
        start_idx = int(curr_video_clip[0].split(os.path.sep)[-1].split(".")[0].split("_")[-1])
        keyframe_idx = int(curr_gt.split(os.path.sep)[-1].split(".")[0].split("_")[-1])
        last_frame_idx = int(curr_video_clip[-1].split(os.path.sep)[-1].split(".")[0].split("_")[-1])

        curr_video_name = curr_video_clip[0].split(os.path.sep)[0]

        assert last_frame_idx == keyframe_idx, \
            f"Current file {curr_video_name} got an error: ground truth is mismatch to the data."

        # getting labels of the current video clip
        label_file = os.path.join(self.label_dir, curr_video_name + ".csv")

        # getting video clips of the current one and previous one
        curr_img_list = self.read_videos(video_clip=curr_video_clip)

        prev_imgs_list = []
        for prev_video_clip in prev_videos_clips:
            prev_img_list = self.read_videos(video_clip=prev_video_clip)
            prev_imgs_list.append(prev_img_list)

        boxes, labels = self.get_annotations(
            label_file, keyframe_idx, img_size=curr_img_list[0].size[0:2])

        original_boxes = np.copy(boxes)
        # original_images = np.copy(np.array(img_list[-1]))

        curr_tensor_list = []
        prev_total_list = []

        if self.transform:
            for idx, img in enumerate(curr_img_list):
                img = np.array(img)
                try:
                    if idx == len(curr_img_list) - 1:
                        img, boxes, labels = self.transform(img, original_boxes, labels)
                    else:
                        img, _, _ = self.transform(img, boxes, labels)
                except IndexError:
                    print("[E] Current data error!")

                curr_tensor_list.append(img)

            for idx, img_list in enumerate(prev_imgs_list):
                prev_tensor_list = []
                for img in img_list:
                    img = np.array(img)
                    img, _, _ = self.prev_transform(img, None, None)
                    prev_tensor_list.append(img)
                prev_total_list.append(torch.stack(prev_tensor_list))

        # plot_img_for_debug(original_images, original_boxes, labels)
        curr_tensor = torch.stack(curr_tensor_list)
        prev_tensor = torch.stack(prev_total_list)

        data = Container(
            prev=prev_tensor,
            curr=curr_tensor
        )

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        targets = Container(
            boxes=boxes,
            labels=labels,
        )

        return data, targets, 0

    def __len__(self):
        return len(self.training_label_list)

    def allocate_consecutive_samples(self):
        img_names_list, gt_list = self._allocate_samples_no_overlap()
        img_names_pairs_list = []
        new_gt_list = []
        for i in range(len(gt_list) - 1):
            prev_images_list = img_names_list[i:(i + self.sequence_num)]
            prev_videos_names = []
            for prev_images in prev_images_list:
                prev_video_name = prev_images[0].split(os.path.sep)[0]
                prev_videos_names.append(prev_video_name)

            prev_videos_names_set = set(prev_videos_names)
            if len(prev_videos_names_set) != 1:
                continue

            prev_video_name = prev_videos_names[0]
            try:
                curr_images = img_names_list[i + self.sequence_num]
            except IndexError:
                break
            curr_video_name = curr_images[0].split(os.path.sep)[0]

            if prev_video_name == curr_video_name:
                curr_gt = gt_list[i + self.sequence_num]

                # TODO
                img_names_pairs_list.append([prev_images_list, curr_images])
                new_gt_list.append(curr_gt)

        return img_names_pairs_list, new_gt_list

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

    def _allocate_samples_no_overlap(self):
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

            for idx in range(0, len(image_names), self.frame_len):
                image_sample_names = image_names[idx:(idx + self.frame_len)]
                key_idx = int(image_sample_names[-1].split(".")[0].split("_")[-1])
                if key_idx in keyframe_idxes_list and len(image_sample_names) == self.frame_len:
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

                boxes.append(np.array(curr_box).astype(np.float64))
                labels.append(self.class_map[label_info[10]])

        assert len(labels), f"[E] File {label_file} labels at index {keyframe_idx} should not be empty."

        if len(boxes) != len(labels):
            print("[E] The length of boxes and labels should be the same.")
            raise ValueError

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
        "../../../configs/resnet/dark/darknet_ssd320_dark_tma_pan.yaml")
    cfg.freeze()

    transform = build_transforms(cfg, True)
    prev_transform = build_prev_clip_transform(cfg)
    target_transform = build_target_transform(cfg)

    dataset = DarkVIDDataset(
        data_dir="/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/Training-Data/20230308/images/train",
        label_dir="/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/Training-Data/20230308/labels",
        frame_len=3,
        transform=transform,
        prev_transform=prev_transform,
        target_transform=target_transform)

    print(len(dataset))
    for j, batch in enumerate(dataset):
        print(j)
        data = batch[0]
        prev = data["prev"]
        curr = data["curr"]
        assert prev.shape[0] == 2, f"[E] Previous clip error, expected 2, got {prev.shape[0]}"
        assert curr.shape[0] == 3, f"[E] Current clip error, expected 3, got {curr.shape[0]}"

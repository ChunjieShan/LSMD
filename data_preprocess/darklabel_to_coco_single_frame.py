#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 将DarkLabel生成的目标追踪label文件（csv格式），转化为coco数据集用的json格式annotations
# Author: SEU-BME-LBMD-chl
import filecmp
import os
from tkinter import Label
import cv2 as cv
import copy
import math
import json
import random
import numpy as np
import pandas as pd
import os.path as path


# Global Constants
# 数据处理输入输出参数

class Label2COCO(object):
    def __init__(self,
                 video_path: str,
                 anno_path: str,
                 dataset_dir: str,
                 label_names: list,
                 newimgsize: int = None,
                 overwrite: bool = True):
        """DarkLabel format to COCO format

        Args:
            video_path (str): original video path
            anno_path (str): csv format annotation path
            dataset_dir (str): destination path
            newimgsize (int, optional): shape image will be resized to. Defaults to None.
            overwrite (bool, optional): overwritten or not. Defaults to True.
        """
        self.images = []
        self.categories = []
        self.annotations = []
        self.class_names = label_names
        self.class_ids = {}

        for i in range(len(self.class_names)):
            self.class_ids[self.class_names[i]] = i + 1

        self.total_videos_coco = []
        self.dataset_out_dir = dataset_dir
        self.anno_path = anno_path
        self.video_path_list, self.anno_info = self._get_video_anno_path_list(video_path, anno_path)
        self.dataset_info_dict = self.split_dataset(0.8)

        self.anno_out_dir = os.path.join(dataset_dir, 'annotations')
        if not os.path.exists(self.anno_out_dir):
            os.mkdir(self.anno_out_dir)

        self.overwrite = overwrite
        self.newimgsize = newimgsize

    def label_and_img_to_coco(self):
        for target, video_path_list in self.dataset_info_dict.items():
            coco = {}
            annos = []
            videos = []
            img_idx = 0
            ann_idx = 0
            ann_class_names = []

            for i, video_path in enumerate(video_path_list):
                # 提示信息
                print('target:{}, processing the {}th label file {}'.format(
                    target, i + 1, os.path.basename(video_path)))
                video_file_name = video_path.split(os.path.sep)[-1].split(".avi")[0]  # get file name without suffix.

                img_output_path = os.path.join(self.dataset_out_dir, target, video_file_name)
                if not os.path.exists(img_output_path):
                    os.mkdir(img_output_path)

                # 读取视频数据
                cap = cv.VideoCapture(video_path)
                img_list_bgr = []
                counter = 1
                while True:
                    # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
                    success, data = cap.read()
                    if not success:
                        break
                    img_bgr = copy.deepcopy(data)
                    img_name = "img_{:05d}.jpg".format(counter)
                    img_h, img_w, _ = data.shape
                    if self.newimgsize:
                        img_bgr = cv.resize(img_bgr, (self.newimgsize, self.newimgsize))
                    cv.imwrite(os.path.join(img_output_path, img_name), img_bgr)
                    img_list_bgr.append(img_bgr)
                    counter += 1
                cap.release()
                data_bgr = copy.deepcopy(np.array(img_list_bgr))

                label_path = os.path.join(self.anno_path, video_file_name + ".csv")
                if os.path.exists(label_path):
                    if str(label_path).endswith(".csv"):
                        label_data = pd.read_csv(label_path, header=None)
                        # [ fn, id, x1, y1, w, h, c=-1, c=-1, c=-1, c=-1, cname]
                        for row in range(label_data.shape[0]):
                            single_label = label_data.values[row, :]
                            idx = single_label[0]
                            img = data_bgr[idx]
                            # img_h, img_w, _ = img.shape
                            if self.newimgsize is None:
                                object_box = single_label[2:6]  # box信息
                            else:
                                resize_rate = [float(self.newimgsize) / img_w, float(self.newimgsize) / img_h]
                                box = single_label[2:6]  # box信息
                                # [x,y,w,h]
                                object_box = [float(int(box[0] * resize_rate[0])), float(int(box[1] * resize_rate[1])),
                                              float(int(box[2] * resize_rate[0])), float(int(box[3] * resize_rate[1]))]

                            class_name = str(single_label[-1])
                            ann_class_names.append(class_name)
                            annos.append(self._get_annotations(
                                box=object_box, image_id=img_idx, ann_id=ann_idx, class_name=class_name))

                            ann_idx += 1

                        videos.append(self._get_images(
                            filename=video_file_name, height=self.newimgsize, width=self.newimgsize, image_id=img_idx))

                    else:
                        annos.append([])
                        videos.append([])

                img_idx += 1

            if len(annos):
                coco["images"] = videos
                coco["annotations"] = annos

                ann_class_names = list(set(ann_class_names))
                if len(self.categories):
                    self.categories.clear()

                for ann_class_name in ann_class_names:
                    self.categories.append(self._get_categories(
                        name=ann_class_name, class_id=self.class_ids[ann_class_name]))
                coco["categories"] = self.categories
                instances = json.dumps(coco)

                anno_save_root_dir = os.path.join(self.anno_out_dir, target)
                if not os.path.exists(anno_save_root_dir):
                    os.mkdir(anno_save_root_dir)

                with open(os.path.join(anno_save_root_dir, 'instances_{}.json'.format(target)), 'w') as f:
                    f.write(instances)

            # else:
            #     os.rmdir(img_output_path)

    def save_anno_and_imgs(self):
        # 记录images、annotations、categories信息，以及解析视频并写入图像
        self.label_and_img_to_coco()

    @staticmethod
    def _get_video_anno_path_list(video_path, anno_path):
        video_path_list = []
        anno_path_list = []
        anno_info = {}

        for root, dirs, files in os.walk(anno_path):
            for file in files:
                single_anno_path = os.path.join(root, file)
                anno_path_list.append(single_anno_path)
                class_name = single_anno_path.split(os.path.sep)[-3]

                file_name = file.split('.csv')[0]

                if file_name not in anno_info.keys():
                    anno_info[file_name] = [class_name]
                else:
                    anno_info[file_name].append(class_name)

        for root, dirs, files in os.walk(video_path):
            for file in files:
                if file.split('.avi')[0] in anno_info.keys():
                    single_video_path = os.path.join(root, file)
                    video_path_list.append(single_video_path)

        return video_path_list, anno_info

    def split_dataset(self, train_percent):
        dataset_target_dict = {}
        train_len = math.floor(len(self.video_path_list) * train_percent)
        random.shuffle(self.video_path_list)

        train_dir = os.path.join(self.dataset_out_dir, 'train')
        val_dir = os.path.join(self.dataset_out_dir, 'val')
        test_dir = os.path.join(self.dataset_out_dir, 'test')

        if not os.path.exists(self.dataset_out_dir):
            os.mkdir(self.dataset_out_dir)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        train_video_list = []
        val_video_list = []

        for _video_path in self.video_path_list[0:train_len]:
            train_video_list.append(_video_path)

        for _video_path in self.video_path_list[train_len:]:
            val_video_list.append(_video_path)

        dataset_target_dict['train'] = train_video_list
        dataset_target_dict['val'] = val_video_list
        dataset_target_dict['test'] = val_video_list

        return dataset_target_dict

    def _get_images(self, filename, height, width, image_id):
        image = {}
        image["height"] = height
        image['width'] = width
        image["id"] = image_id
        # 文件名加后缀
        image["file_name"] = filename
        # print(image)
        return image

    def _get_annotations(self, box, image_id, ann_id, class_name):
        annotation = {}
        w, h = box[2], box[3]
        area = w * h
        annotation['segmentation'] = [[]]
        annotation['iscrowd'] = 0
        # 第几张图像，从0开始
        annotation['image_id'] = image_id
        annotation['bbox'] = box
        annotation['area'] = float(area)
        # category_id=0
        annotation['category_id'] = self.class_ids[class_name]
        # 第几个标注，从0开始
        annotation['id'] = ann_id
        # print(annotation)
        return annotation

    def _get_categories(self, name, class_id):
        category = {}
        category["supercategory"] = "object"
        # id=0
        category['id'] = class_id
        # name=1
        category['name'] = name
        # print(category)
        return category


if __name__ == '__main__':
    ROOTDIR = "/home/scj/Code/Data/1.Echocardiogram/2.Data/PSAXMV/2.Labelled-Data/Object-Detection/single_frame_dataset"
    ROOTDIR_LABEL = os.path.join(ROOTDIR, "labels")
    ROOTDIR_VIDEO = os.path.join(ROOTDIR, "video_data")
    ROOTDIR_DATASET = os.path.join(ROOTDIR, "Dataset_ssd_v3/")
    # OVERWRITE = False  # 是否覆盖原有样本
    OVERWRITE = True  # 是否覆盖原有样本
    NEWSIZE = 320  # 图像缩放后尺寸

    label_names = ["heart", "LV", "MV", "RV", "LVF", "LVS", "LVB", "LVL"]
    # label_names = ["cross", "vertical"]

    # dataset = Label2CoCo(dataset_dir=ROOTDIR_DATASET, sample_out_dir=ROOTDIR_DATASET,
    #                         anno_dir='annotations', newimgsize=NEWSIZE, overwrite=True)
    dataset = Label2COCO(ROOTDIR_VIDEO, ROOTDIR_LABEL, ROOTDIR_DATASET, label_names, 320, True)
    dataset.save_anno_and_imgs()

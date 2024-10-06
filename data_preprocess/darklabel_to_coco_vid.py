#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 将DarkLabel生成的目标追踪label文件（csv格式），转化为coco数据集用的json格式annotations
# Author: SEU-BME-LBMD-chl
import filecmp
import os
from tkinter import Label
import cv2
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
        self.class_names = os.listdir(anno_path)
        self.class_ids = {}

        for i in range(len(self.class_names)):
            self.class_ids[self.class_names[i]] = i + 1 

        self.total_videos_coco = []
        self.dataset_out_dir = dataset_dir
        self.anno_path = anno_path
        self.video_path_list, self.anno_info = self._get_video_anno_path_list(video_path, anno_path)
        self.dataset_target_list = self.split_dataset(0.8)

        self.anno_out_dir = os.path.join(dataset_dir, 'annotations')
        if not os.path.exists(self.anno_out_dir): os.mkdir(self.anno_out_dir)

        self.overwrite = overwrite
        self.newimgsize = newimgsize

    def label_and_img_to_coco(self):
        for i in range(len(self.dataset_target_list)):
            coco = {}
            curr_imgs = []
            curr_annos = []
            img_idx = 0
            ann_idx = 0
            ann_class_names = []

            dataset_info = self.dataset_target_list[i]
            # 提示信息
            print('target:{}, processing the {}th label file {}'.format(
                dataset_info['target'], i + 1, os.path.basename(self.video_path_list[i])))

            target = dataset_info['target']
            video_path = dataset_info['video_path']
            video_file_name = video_path.split(os.path.sep)[-1].split(".avi")[0]  # get file name without suffix.

            img_output_path = os.path.join(self.dataset_out_dir, target, video_file_name)
            if not os.path.exists(img_output_path):
                os.mkdir(img_output_path)

            # 读取视频数据
            cap = cv2.VideoCapture(video_path)
            img_list_bgr = []
            while True:
                # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
                success, data = cap.read()
                if not success:
                    break
                img_bgr = copy.deepcopy(data)
                img_list_bgr.append(img_bgr)
            cap.release()
            data_bgr = copy.deepcopy(np.array(img_list_bgr))

            # 读取多种标注数据
            label_classes = self.anno_info[video_file_name]
            label_datas = []
            label_data_idxs = []
            for label_class in label_classes:
                label_path = os.path.join(self.anno_path, label_class, "video_label", video_file_name + ".csv")
                # 对应位置存在csv格式label文件
                if os.path.exists(label_path):
                    if str(label_path).endswith('.csv'):
                        label = pd.read_csv(label_path)
                        # DarkLabel生成的目标追踪label文件（csv格式）各列分别为：
                        # [ fn, id, x1, y1, w, h, c=-1, c=-1, c=-1, c=-1, cname]
                        label_datas.append(copy.deepcopy(label.values[:, :].tolist()))
                        # 帧索引
                        label_idx = copy.deepcopy(label.values[:, 0].tolist())
                        label_data_idxs.append(label_idx)

                # 对应位置不存在csv格式label文件
                else:
                    label_datas.append([])
                    label_data_idxs.append([])

            for frame_idx in range(data_bgr.shape[0]):
                # 视频图像信息
                img_data = copy.deepcopy(data_bgr[frame_idx, :, :, :])
                img_h, img_w = img_data.shape[:-1]
                img_new_h, img_new_w = img_data.shape[:-1]
                img_name = 'img_{:05d}.jpg'.format(frame_idx)

                # 对视频的第frame_idx帧，每类目标分别处理
                empty = True
                object_boxs = []
                object_box = None
                label_data = None

                for class_i in range(len(label_datas)):
                    if len(label_data_idxs[class_i]) < 1 or len(label_datas[class_i]) < 1:
                        continue

                    # 对视频的第frame_idx帧，处理第class_i类目标的信息
                    if frame_idx in label_data_idxs[class_i]:
                        label_data_idx = label_data_idxs[class_i].index(frame_idx)
                        label_data = copy.deepcopy(label_datas[class_i][label_data_idx])
                        # 标注信息
                        if self.newimgsize is None:
                            object_box = label_data[2:6]  # box信息
                        else:
                            resize_rate = [float(self.newimgsize) / img_w, float(self.newimgsize) / img_h]
                            box = label_data[2:6]  # box信息
                            # [x,y,w,h]
                            object_box = [float(int(box[0] * resize_rate[0])), float(int(box[1] * resize_rate[1])),
                                          float(int(box[2] * resize_rate[0])), float(int(box[3] * resize_rate[1]))]
                        class_name = str(label_data[-1])
                        object_boxs.append((class_name, object_box))

                        # json文件信息

                        if len(object_box):
                            curr_annos.append(self._get_annotations(
                                box=object_box, image_id=img_idx, ann_id=ann_idx, class_name=class_name))

                        else:
                            print("Empty Box!")
                            exit(-1)

                        ann_idx += 1
                        empty = False
                        ann_class_names.append(str(label_data[-1]))

                if not empty:
                    if self.newimgsize is not None:
                        img_new_h, img_new_w = self.newimgsize, self.newimgsize

                    curr_imgs.append(self._get_images(
                        filename=img_name, height=img_new_h, width=img_new_w, image_id=img_idx))

                    img_path = os.path.join(img_output_path, img_name)
                    print("Img path: ", img_path)

                    if self.overwrite or (not os.path.exists(img_path)):
                        if self.newimgsize is not None:
                            img_data = cv2.resize(img_data, dsize=(self.newimgsize, self.newimgsize),
                                                    interpolation=cv2.INTER_LINEAR)

                        img_show = copy.deepcopy(img_data)
                        for classname, object_box in object_boxs:
                            # drawing boxes on screen for debugging
                            cv2.rectangle(img_show, (int(object_box[0]), int(object_box[1])),
                                            (int(object_box[0] + object_box[2]), int(object_box[1] + object_box[3])),
                                            (0, 255, 0), 1)

                        # cv2.imshow('Sample showing for debugging', img_show)
                        # cv2.waitKey(10)
                        cv2.imwrite(img_path, img_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        img_idx += 1

            if len(curr_annos):
                coco["images"] = curr_imgs
                
                coco["annotations"] = curr_annos

                ann_class_names = list(set(ann_class_names))
                if len(self.categories):
                    self.categories.clear()

                for ann_class_name in ann_class_names:
                    self.categories.append(self._get_categories(
                        name=ann_class_name, class_id=self.class_ids[ann_class_name]))
                coco["categories"] = self.categories
                instances = json.dumps(coco)

                anno_save_root_dir = os.path.join(self.anno_out_dir, target)
                if not os.path.exists(anno_save_root_dir): os.mkdir(anno_save_root_dir)

                with open(os.path.join(anno_save_root_dir, video_file_name + '.json'), 'w') as f:
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
        dataset_target_list = []
        train_len = math.floor(len(self.video_path_list) * train_percent)
        random.shuffle(self.video_path_list)

        train_dir = os.path.join(self.dataset_out_dir, 'train')
        val_dir = os.path.join(self.dataset_out_dir, 'val')
        test_dir = os.path.join(self.dataset_out_dir, 'test')

        if not os.path.exists(train_dir): os.mkdir(train_dir)
        if not os.path.exists(val_dir): os.mkdir(val_dir)
        if not os.path.exists(test_dir): os.mkdir(test_dir)

        for _video_path in self.video_path_list[0:train_len]:
            dataset_target_list.append(
                dict(
                    target = "train",
                    video_path = _video_path
                )
            )

        for _video_path in self.video_path_list[train_len:]:
            dataset_target_list.append(
                dict(
                    target = "val",
                    video_path = _video_path
                )
            )

        return dataset_target_list

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
    ROOTDIR = '/home/scj/Projects/Code/Data/PSAX-A/PSAX_OD'
    ROOTDIR_LABEL = os.path.join(ROOTDIR, "PSAXA_selected_label/object_detect/")
    ROOTDIR_VIDEO = os.path.join(ROOTDIR, 'PSAXA_selected_video/video_data/')
    ROOTDIR_DATASET = os.path.join(ROOTDIR, "Dataset_ssd_v9/")
    # OVERWRITE = False  # 是否覆盖原有样本
    OVERWRITE = True  # 是否覆盖原有样本
    NEWSIZE = 300  # 图像缩放后尺寸

    # label_names = ["", "right"]

    # dataset = Label2CoCo(dataset_dir=ROOTDIR_DATASET, sample_out_dir=ROOTDIR_DATASET,
    #                         anno_dir='annotations', newimgsize=NEWSIZE, overwrite=True)
    dataset = Label2COCO(ROOTDIR_VIDEO, ROOTDIR_LABEL, ROOTDIR_DATASET, 300, True)
    dataset.save_anno_and_imgs()

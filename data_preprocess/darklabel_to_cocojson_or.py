#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 将DarkLabel生成的目标追踪label文件（csv格式），转化为coco数据集用的json格式annotations
# Author: SEU-BME-LBMD-chl
import os
import cv2
import copy
import math
import json
import random
import numpy as np
import pandas as pd
import os.path as path
import xml.dom.minidom as mini_dom
from xml.dom.minidom import parse

# Global Constants
# 数据处理输入输出参数
ROOTDIR = '/home/scj/Projects/Code/Data/PSAX-MV/single_frame_dataset'
ROOTDIR_LABEL = os.path.join(ROOTDIR, "PSAXA_selected_label_up_right/object_detect/")
ROOTDIR_VIDEO = os.path.join(ROOTDIR, 'PSAXA_selected_video/video_data/')
# ROOTDIR = r'E:\Data\PSAX-A\dirty_data'
# ROOTDIR_LABEL = os.path.join(ROOTDIR, "object_detect_label")
# ROOTDIR_VIDEO = os.path.join(ROOTDIR, 'video_data\\')
ROOTDIR_DATASET = os.path.join(ROOTDIR, "Dataset_ssd_v9/")
# ROOTDIR = r'E:\Data\PSAX-A\dirty_data'
# ROOTDIR_LABEL = os.path.join(ROOTDIR, "object_detect_label")
# ROOTDIR_VIDEO = os.path.join(ROOTDIR, 'video_data\\')
# OVERWRITE = False  # 是否覆盖原有样本
OVERWRITE = True  # 是否覆盖原有样本
NEWSIZE = 300  # 图像缩放后尺寸


class Label2CoCo(object):
    def __init__(self, paths, dataset_dir, sample_dir, anno_dir, video_paths=None, newimgsize=None, overwrite=True):
        self.images = []
        self.categories = []
        self.annotations = []
        # 可根据情况设置类别，这里只设置了一类
        # self.class_id = 4
        # self.class_ids = {'A': 1,'IAS': 2,'LA': 3,'PV': 4}
        # self.class_mapping = {'DSJ':'A','PSAXGV_IAS':'IAS','PSAXGV_LA':'LA','PSAXGV_PV':'PV'}
        # self.color_mapping = [(0,0,0),(0,255,255),(255,0,255),(255,255,0),(128,192,255),(192,255,128),(255,128,192)]
        self.class_id = 2
        # self.class_ids = {'A':1,'AV':2,'LA':3,'PA':4,'RA':5,'RVOT':6,'HEART':7}
        # self.class_ids = {'HEART': 1, 'IAS': 2, 'IVS': 3, 'LA': 4, 'LV': 5, 'MV': 6, 'RA': 7, 'RV': 8, 'TV': 9}
        # self.class_ids = {"PSAXA_HEART": 1, "PSAXA_LV": 2, "PSAXA_MUSCLE": 3, "up": 4, "right": 5}
        # self.class_mapping = {"PSAXA_HEART": "PSAXA_HEART", "PSAXA_LV": "PSAXA_LV", "PSAXA_MUSCLE": "PSAXA_MUSCLE",
        #                       "up": "up", "right": "right"}
        self.class_ids = {"up": 1, "right": 2}
        self.class_mapping = {"up": "up", "right": "right"}
        # self.color_mapping = [(0, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 192, 255), (192, 255, 128),
        #                       (255, 128, 192), (255, 233, 192), (200, 233, 192), (255, 150, 192)]  # 比类别数多1
        self.color_mapping = [(0, 0, 0), (0, 255, 255), (0, 255, 0), (0, 0, 255)]

        self.coco = {}
        #
        self.overwrite = overwrite
        self.newimgsize = newimgsize
        #
        self.video_paths = [path_i[0] for path_i in paths]
        self.label_paths = [path_i[1] for path_i in paths]
        #
        self.dataset_outdir = dataset_dir
        self.target = sample_dir.split(os.path.sep)[0].split(os.path.sep)[0]
        self.anno_outdir = os.path.join(self.dataset_outdir, anno_dir)
        self.sample_outdir = os.path.join(self.dataset_outdir, self.target)
        if not os.path.exists(self.anno_outdir): os.makedirs(self.anno_outdir)
        if not os.path.exists(self.sample_outdir): os.makedirs(self.sample_outdir)

    def label_and_img_to_coco(self):
        img_idx = 0
        ann_idx = 0
        ann_class_names = []
        for i in range(len(self.video_paths)):
            # 提示信息
            print('target:{}, processing the {}th label file {}'.format(
                self.target, i + 1, os.path.basename(self.video_paths[i])))
            # 读取视频数据
            cap = cv2.VideoCapture(self.video_paths[i])
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
            label_datas = []
            label_data_idxs = []
            for label_i in range(len(self.label_paths[i])):
                label_path = copy.deepcopy(self.label_paths[i][label_i])
                # 对应位置存在csv格式label文件
                if os.path.exists(label_path):
                    if str(label_path).endswith('.csv'):
                        label = pd.read_csv(label_path)
                        # DarkLabel生成的目标追踪label文件（csv格式）各行分别为：
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
                img_name = '{}_img_{}.jpg'.format(self.target, img_idx)

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
                            #
                            box = label_data[2:6]  # box信息
                            # [x,y,w,h]
                            object_box = [float(int(box[0] * resize_rate[0])), float(int(box[1] * resize_rate[1])),
                                          float(int(box[2] * resize_rate[0])), float(int(box[3] * resize_rate[1]))]
                        class_name = self.class_mapping[str(label_data[-1])]
                        object_boxs.append((class_name, object_box))
                        # json文件信息
                        self.annotations.append(self._get_annotations(
                            box=object_box, image_id=img_idx, ann_id=ann_idx, class_name=class_name))
                        ann_idx += 1
                        empty = False
                        ann_class_names.append(self.class_mapping[str(label_data[-1])])

                if not empty:
                    if self.newimgsize is not None:
                        img_new_h, img_new_w = self.newimgsize, self.newimgsize
                    self.images.append(self._get_images(
                        filename=img_name, height=img_new_h, width=img_new_w, image_id=img_idx))
                    img_path = os.path.join(self.sample_outdir, img_name)
                    if self.overwrite or (not os.path.exists(img_path)):
                        if self.newimgsize is not None:
                            img_data = cv2.resize(img_data, dsize=(self.newimgsize, self.newimgsize),
                                                  interpolation=cv2.INTER_LINEAR)
                        img_show = copy.deepcopy(img_data)
                        for classname, object_box in object_boxs:
                            # cv2.rectangle(img_show,(x1,y1),(x2,y2),...)
                            cv2.rectangle(img_show, (int(object_box[0]), int(object_box[1])),
                                          (int(object_box[0] + object_box[2]), int(object_box[1] + object_box[3])),
                                          self.color_mapping[self.class_ids[classname]], 1)
                        cv2.imshow('sample_show', img_show)
                        cv2.waitKey(10)
                        cv2.imwrite(img_path, img_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    img_idx += 1
        self.coco["images"] = self.images
        self.coco["annotations"] = self.annotations

        ann_class_names = list(set(ann_class_names))
        for ann_class_name in ann_class_names:
            self.categories.append(self._get_categories(
                name=ann_class_name, class_id=self.class_ids[ann_class_name]))
        self.coco["categories"] = self.categories

    def save_json(self):
        # 记录images、annotations、categories信息，以及解析视频并写入图像
        self.label_and_img_to_coco()
        label_dict = self.coco
        instances = json.dumps(label_dict)
        f = open(os.path.join(self.dataset_outdir + 'annotations/instances_{}.json'.format(self.target)), 'w')
        f.write(instances)

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


###############################################################################
# Helper Functions
###############################################################################
'''根据设定参数划分数据集(train,val,test)'''


def split_dataset(datadir, labeldir, datasetdir, label_names, dataend='.avi', labelend='.csv',
                  trainval_percent=0.8, train_percent=0.8, seed=0):
    """
    :param labelfiledir: DarkLabel生成的MOT任务注释csv文档所在路径
    :param datasetdir: coco数据集对应根目录
    :return: train,val,test文件列表
    """
    # 由于可能出现数据数目多于标注数目的情况，因此以标注文件名为准
    data_lists = get_filepaths(datadir, suffix=dataend, onlypath=True)
    # 多种不同的label csv
    label_dirs = [os.path.join(labeldir, label_name, 'video_label') for label_name in label_names]
    # 求各类标注文件与视频文件名的交集
    label_lists = [
        dira_with_dirb(label_dir, datadir, fileflag='or', ignore_strs=['.csv', '.avi'], suffix=['.csv', '.avi'])
        for label_dir in label_dirs]
    # label_lists = [pathsa_with_pathsb(label_dir, datadir, fileflag='or', ignore_strs=['.csv','.avi'], suffix=['.csv','.avi'])
    #                for label_dir in label_dirs]

    # [class * [N]] ---> [[class] *N]
    datas = []
    labels = []
    data_lists_copy = copy.deepcopy(data_lists)
    for i in range(len(label_lists[0])):
        label_temp = []
        for j in range(len(label_lists)):
            label_temp.append(label_lists[j][i])
        labels.append(label_temp)
        # 注意求得交集后，顺序可能存在差异，因此需要进行再次排序，与视频数据名称一一对应
        filename = copy.deepcopy(os.path.basename(label_lists[0][i])).rsplit('.', 1)[0]
        for datapath in data_lists_copy:
            if datapath.find(filename) > -1:
                datas.append(datapath)
                data_lists_copy.remove(datapath)
                break

    str_a = datas[0]
    str_b = labels[0]
    str_diff = strDiff(str_a, str_b)

    # 划分数据集
    random.seed(seed)
    num = len(labels)
    list_index = range(num)
    len_train = int(num * trainval_percent * train_percent)
    len_val = int(num * trainval_percent) - len_train
    len_test = len(labels) - int(num * trainval_percent)
    train_idx = random.sample(list_index, len_train)
    val_idx = random.sample(list_index, len_val)
    test_idx = random.sample(list_index, len_test)
    train = [[datas[i], labels[i]] for i in train_idx]
    val = [[datas[i], labels[i]] for i in val_idx]
    test = [[datas[i], labels[i]] for i in test_idx]
    # 新建数据集各文件夹
    dataset_dirs = ['train', 'val', 'test']
    for dataset_dir in dataset_dirs:
        outdir = os.path.join(datasetdir, dataset_dir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    return {'train': train, 'val': val, 'test': test}


'''获取根目录下各文件名及文件完整路径(path+filename)'''


def get_filepaths(indir, prefix='', suffix='', size_range=[], onlypath=False):
    """
    :param indir: 文件根目录
    :param prefix: 文件前缀（如"CT."），默认不指定
    :param suffix: 文件后缀（如"*.dcm"），默认不指定
    :param size_range: 指定文件尺寸（[min,max]单位bytes）
    :param onlypath: 返回时仅返回根目录下各文件完整路径
    :return: 根目录下指定后缀类型文件的各文件及文件完整路径（list类型）
    """
    # 检查传入参数
    assert len(size_range) == 0 or len(size_range) == 2, "len(size_range) = 0 or 2!"
    assert (not suffix) or type(suffix) == str or type(suffix) == list, "suffix should be None, str or list!"
    refilenames = []  # 存放各文件名
    refiledirs = []  # 存放各文件路径
    refilepaths = []  # 存放各文件完整路径(path+filename)
    rootdir = copy.deepcopy(indir)
    for root_dirs, file_dirs, filenames in os.walk(rootdir):
        for filename in filenames:
            # 非指定文件类型则跳过
            if suffix:
                if (type(suffix) == str) and (not filename.endswith(suffix)):
                    continue
                if type(suffix) == list:
                    str_temp = '.{}'.format(filename.rsplit(sep='.', maxsplit=1)[-1])
                    if not ((str_temp in suffix) or (str_temp[1:] in suffix)):
                        continue
            if prefix and (not filename.startswith(prefix)):
                continue
            # 非指定文件尺寸则跳过
            if len(size_range) > 0:
                file_size = os.path.getsize(os.path.join(root_dirs, filename))
                if ((file_size < size_range[0]) or (file_size > size_range[1])):
                    continue
            refilepaths.append(os.path.join(root_dirs, filename))
            refiledirs.append(os.path.join(root_dirs))
            refilenames.append(os.path.join(filename))

    if not onlypath:
        return refilepaths, refiledirs, refilenames
    else:
        return refilepaths


def dira_with_dirb(dir_a, dir_b, fileflag='same', ignore_strs=None, suffix=['.csv', '.avi']):
    file_paths_a, _, filenames_a = get_filepaths(dir_a, suffix=suffix[0], onlypath=False)
    file_paths_b, _, filenames_b = get_filepaths(dir_b, suffix=suffix[1], onlypath=False)

    # 部分时候文件名存在有规律的差异，忽略这种差异
    ignore_dict = dict()
    if isinstance(ignore_strs, str):
        ignore_dict[ignore_strs] = ''
    elif isinstance(ignore_strs, list):
        for ignore_str in ignore_strs:
            ignore_dict[ignore_str] = ''
    else:
        ignore_dict['.'] = '.'
    for key, val in ignore_dict.items():
        filenames_a = [filename.replace(key, val) for filename in filenames_a]
        filenames_b = [filename.replace(key, val) for filename in filenames_b]

    file_list = []
    if fileflag == 'same':
        # 取交集
        same_list = list(set(filenames_a).intersection(set(filenames_b)))
        for file_i in same_list:
            file_idx = filenames_a.index(file_i)
            file_list.append(file_paths_a[file_idx])
    elif fileflag == 'or':
        # 取并集
        same_list = list(set(filenames_a).union(set(filenames_b)))
        for file_i in same_list:
            if file_i in filenames_a:
                file_idx = filenames_a.index(file_i)
                file_list.append(file_paths_a[file_idx])
            else:
                file_idx = filenames_b.index(file_i)
                file_list.append(file_paths_b[file_idx])
    elif fileflag == 'adiffb':
        # 获取a有而b没有的
        diff_list = list(set(filenames_a).difference(set(filenames_b)))
        for file_i in diff_list:
            file_idx = filenames_a.index(file_i)
            file_list.append(file_paths_a[file_idx])
    elif fileflag == 'bdiffa':
        # 获取b有而a没有的
        diff_list = list(set(filenames_b).difference(set(filenames_a)))
        for file_i in diff_list:
            file_idx = filenames_b.index(file_i)
            file_list.append(file_paths_b[file_idx])

    return file_list


def pathsa_with_pathsb(paths_a, paths_b, fileflag='same', ignore_strs=None, suffix=['.csv', '.avi']):
    file_paths_a = copy.deepcopy(paths_a) if isinstance(paths_a, list) else get_filepaths(paths_a, suffix=suffix[0],
                                                                                          onlypath=True)
    file_paths_b = copy.deepcopy(paths_b) if isinstance(paths_b, list) else get_filepaths(paths_b, suffix=suffix[1],
                                                                                          onlypath=True)
    filenames_a = [os.path.basename(file_path_a) for file_path_a in file_paths_a]
    filenames_b = [os.path.basename(file_path_b) for file_path_b in file_paths_b]

    # 部分时候文件名存在有规律的差异，忽略这种差异
    ignore_dict = dict()
    if isinstance(ignore_strs, str):
        ignore_dict[ignore_strs] = ''
    elif isinstance(ignore_strs, list):
        for ignore_str in ignore_strs:
            ignore_dict[ignore_str] = ''
    else:
        ignore_dict['.'] = '.'
    for key, val in ignore_dict.items():
        filenames_a = [filename.replace(key, val) for filename in filenames_a]
        filenames_b = [filename.replace(key, val) for filename in filenames_b]

    file_list = []
    if fileflag == 'same':
        # 取交集
        same_list = list(set(filenames_a).intersection(set(filenames_b)))
        for file_i in same_list:
            file_idx = filenames_a.index(file_i)
            file_list.append(file_paths_a[file_idx])
    elif fileflag == 'or':
        # 取并集
        same_list = list(set(filenames_a).union(set(filenames_b)))
        for file_i in same_list:
            if file_i in filenames_a:
                file_idx = filenames_a.index(file_i)
                file_list.append(file_paths_a[file_idx])
            else:
                file_idx = filenames_b.index(file_i)
                file_list.append(file_paths_b[file_idx])
    elif fileflag == 'adiffb':
        # 获取a有而b没有的
        diff_list = list(set(filenames_a).difference(set(filenames_b)))
        for file_i in diff_list:
            file_idx = filenames_a.index(file_i)
            file_list.append(file_paths_a[file_idx])
    elif fileflag == 'bdiffa':
        # 获取b有而a没有的
        diff_list = list(set(filenames_b).difference(set(filenames_a)))
        for file_i in diff_list:
            file_idx = filenames_b.index(file_i)
            file_list.append(file_paths_b[file_idx])

    return file_list


def strDiff(str1, str2):
    tmp = {index: val for index, val in enumerate(str1) if
           len(str2) <= index or (len(str2) > index and not str2[index] == val)}
    return "".join(tmp.values())


if __name__ == '__main__':
    # label_names = ['A', 'IAS', 'LA', 'PV']
    # label_names = ['A']
    # label_names = ['HEART', 'IAS', 'IVS', 'LA', 'LV', 'MV', 'RA', 'RV', 'TV']
    # label_names = ["PSAXA_HEART", "PSAXA_LV", "PSAXA_MUSCLE", "up", "right"]
    label_names = ["up", "right"]
    dataset = split_dataset(datadir=ROOTDIR_VIDEO, labeldir=ROOTDIR_LABEL, datasetdir=ROOTDIR_DATASET,
                            label_names=label_names, dataend='.avi', labelend='.csv', trainval_percent=0.8,
                            train_percent=0.8, seed=0)

    # 训练集
    for target, paths in dataset.items():
        dataset_i = Label2CoCo(paths=paths, dataset_dir=ROOTDIR_DATASET, sample_dir=target,
                               anno_dir='annotations', newimgsize=NEWSIZE, overwrite=True)
        dataset_i.save_json()

#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 将label文件（json格式），转化为分类数据集用的样本
# Author: SEU-BME-LBMD-chl
import os
import cv2
import copy
import math
import json
import random
import numpy as np

from extern.fileopts import split_filename,get_filepaths


# Global Constants
# 数据处理输入输出参数
ROOTDIR = 'F:/BME_New/Echocardiography_Datas/Quality_Assessment_object_detection/'
# ROOTDIR_LABEL = ROOTDIR + 'PSAX_selected_labelsss/classification/labeljson/'
# ROOTDIR_VIDEO = ROOTDIR + 'PSAX_selected_video_data/'
# ROOTDIR_DATASET = ROOTDIR + 'Dataset_of_ssd_transformer_v1/'

ROOTDIR_LABEL = ROOTDIR + 'PSAX_selected_labelsss/classification/labeljson_ext_test/'
ROOTDIR_VIDEO = ROOTDIR + 'PSAX_selected_video_data/video_data_ext_test/'
ROOTDIR_DATASET = ROOTDIR + 'Dataset_of_ssd_transformer_v1/'
# OVERWRITE = False  # 是否覆盖原有样本
OVERWRITE = True  # 是否覆盖原有样本
NEWSIZE = 300     #图像缩放后尺寸




'''根据设定参数划分数据集(train,val,test)'''
def split_dataset(datadir,labeldir,datasetdir,dataend='.avi',labelend='.json',
                  trainval_percent=0.8,train_percent=0.8,seed=0):
    """
    :param labelfiledir: DarkLabel生成的MOT任务注释csv文档所在路径
    :param datasetdir: coco数据集对应根目录
    :return: train,val,test文件列表
    """
    # 由于可能出现数据数目多于标注数目的情况，因此以标注文件名为准
    datadir = get_filepaths(datadir,suffix=dataend,onlypath=True)
    labels = get_filepaths(labeldir, suffix=labelend, onlypath=True)
    # 划分数据集
    random.seed(seed)
    num = len(labels)
    list_index = range(num)
    len_train = int(num*trainval_percent*train_percent)
    len_val = int(num*trainval_percent) - len_train
    len_test = len(labels) - int(num*trainval_percent)
    train_idx = random.sample(list_index, len_train)
    val_idx = random.sample(list_index, len_val)
    test_idx = random.sample(list_index, len_test)
    train = [[datadir[i],labels[i]] for i in train_idx]
    val = [[datadir[i],labels[i]] for i in val_idx]
    test = [[datadir[i],labels[i]] for i in test_idx]
    # 新建数据集各文件夹
    dataset_dirs = ['train','val','test']
    for dataset_dir in dataset_dirs:
        outdir = os.path.join(datasetdir,dataset_dir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    return {'train':train, 'val':val, 'test':test}


'''写入图像或视频'''
def data_w_label2samples(paths, dataset_dir, sample_dir,newimgsize=None, overwrite=True):
    dataset_outdir = dataset_dir
    target = sample_dir.split('/')[0].split('\\')[0]
    sample_outdir = os.path.join(dataset_outdir, target)
    if not os.path.exists(dataset_outdir): os.makedirs(dataset_outdir)
    if not os.path.exists(sample_outdir): os.makedirs(sample_outdir)

    img_idx = 0
    for i in range(len(paths)):
        video_path = copy.deepcopy(paths[i][0])
        label_path = copy.deepcopy(paths[i][1])

        # 提示信息
        print('target:{}, processing the {}th label file {}'.format(
            target, i + 1, os.path.basename(video_path)))
        # 读取视频数据
        cap = cv2.VideoCapture(video_path)
        img_list_bgr = []
        while True:
            # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
            success, frame = cap.read()
            if not success:
                break
            img_bgr = copy.deepcopy(frame)
            img_list_bgr.append(img_bgr)
        cap.release()
        data = copy.deepcopy(np.array(img_list_bgr))
        # 读取标注数据
        with open(label_path, 'r', encoding='utf8')as fp:
            label_dict = json.load(fp)
        assert int(label_dict['n_frames'])==data.shape[0],"frame should be the same with video.shape[0]"
        labels = copy.deepcopy(label_dict['label'])
        assert len(labels) == data.shape[0], "len(labels) should be the same with video.shape[0]"
        for i in range(data.shape[0]):
            img_data = copy.deepcopy(data[i, ...])
            img_name = '{}_img_{}_{}.jpg'.format(target, img_idx, int(labels[i]))
            img_path = os.path.join(sample_outdir, img_name)
            if overwrite or (not os.path.exists(img_path)):
                if newimgsize is not None:
                    img_data = cv2.resize(img_data, dsize=(newimgsize, newimgsize),interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(img_path, img_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            img_idx += 1




if __name__ == '__main__':
    # dataset = split_dataset(datadir=ROOTDIR_VIDEO,labeldir=ROOTDIR_LABEL,datasetdir=ROOTDIR_DATASET,
    #                         dataend='.avi',labelend='.json',trainval_percent=0.8,train_percent=0.8,seed=0)
    # # 训练集
    # for target, paths in dataset.items():
    #     if len(paths)<1:
    #         continue
    #     data_w_label2samples(paths=paths, dataset_dir=ROOTDIR_DATASET, sample_dir=target,
    #                          newimgsize=NEWSIZE, overwrite=True)

    dataset = split_dataset(datadir=ROOTDIR_VIDEO, labeldir=ROOTDIR_LABEL, datasetdir=ROOTDIR_DATASET,
                            dataend='.avi', labelend='.json', trainval_percent=0.0, train_percent=0.0, seed=0)
    # 训练集
    for target, paths in dataset.items():
        if len(paths) < 1:
            continue
        data_w_label2samples(paths=paths, dataset_dir=ROOTDIR_DATASET, sample_dir=target,
                             newimgsize=NEWSIZE, overwrite=True)
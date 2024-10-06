#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 将超声心动图(*.avi)解析为JPEG图像；并生成n_frames
# Author: SEU-BME-LBMD-zzy,SEU-BME-LBMD-chl,SEU-BME-LBMD-cyc
import os
import cv2
import copy
import math
import shutil
import numpy as np
from skimage import draw,transform
from extern.fileopts import split_filename,split_filepath,get_filedirdict
from extern.get_roi_and_datarange import get_img_roi,get_img_range,create_kernel,get_findContours_renum


# Global Constants
ROOTDIR = 'F:/BME_New/Echocardiography_Datas/Quality_Assessment_keyframe_detection/'
ROOTDIR_VIDEO = ROOTDIR + 'video_data/A4C/'     # 视频文件输入根目录
ROOTDIR_IMAGE = ROOTDIR + 'image_data_processed/'         # 解析图像输出根目录
OVERWRITE = True                                # 是否覆盖原有数据
# OVERWRITE = False                                # 是否覆盖原有数据


# # 使用ffmeg将视频解析为图像，但需要系列配置，弃用该部分功能
# def class_process(rootdir_video, rootdir_img):
#     """
#     :param rootdir_video: 视频文件输入根目录
#     :param rootdir_img: 解析后图像输出根目录
#     """
#     rootdir_v = copy.deepcopy(rootdir_video)
#     if not os.path.isdir(rootdir_v):
#         return
#
#     rootdir_i = copy.deepcopy(rootdir_img)
#     if not os.path.exists(rootdir_i):
#         os.mkdir(rootdir_i)
#
#     for file_name in os.listdir(rootdir_v):
#         # 若文件后缀不为*.avi则直接跳过
#         if not file_name.endswith('.avi'):
#           continue
#         name, ext = os.path.splitext(file_name)
#         outdir_img = os.path.join(rootdir_i, name)
#         video_file_path = os.path.join(rootdir_video, file_name)
#
#         # 使用ffmeg将视频解析为图像，但需要系列配置，弃用该部分功能
#         if os.path.exists(outdir_img):
#             # 若目标文件夹不为空，且选择覆盖原有数据，则直接直接删除目标文件夹
#             if len(os.listdir(outdir_img)) > 0:
#                 if OVERWRITE:
#                     subprocess.call('rm -r \"{}\"'.format(outdir_img), shell=True)
#                     print('remove {}'.format(outdir_img))
#                     os.mkdir(outdir_img)
#                 else:
#                     continue
#         else:
#             os.mkdir(outdir_img)
#         cmd = 'ffmpeg -i \"{}\" -vf scale=-1:-1 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
#         print(cmd)
#         subprocess.call(cmd, shell=True)
#         print('\n')


'''将超声心动图(*.avi)解析为JPEG图像'''
def video2img(rootdir_video, rootdir_img):
    """
    :param rootdir_video: 视频文件输入根目录
    :param rootdir_img: 解析后图像输出根目录
    :return: 超声心动图(*.avi)解析得到的JPEG图像
    """
    # 检查输入路径是否存在，若存在则递归创建输出根目录
    rootdir_v = copy.deepcopy(rootdir_video)
    if not os.path.isdir(rootdir_v): return
    rootdir_i = copy.deepcopy(rootdir_img)
    if not os.path.exists(rootdir_i): os.makedirs(rootdir_i)
    # 获取“各子文件夹-子文件夹内文件”的字典
    filedict_video = get_filedirdict(rootdir_v, suffix='.avi')

    for subdir_v,file_names in filedict_video.items():
        for file_name in file_names:
            # 若文件后缀不为*.avi则直接跳过
            if not file_name.endswith('.avi'):
                continue
            name, ext = os.path.splitext(file_name)
            # 以N种标准切面名称（对应video子文件夹名）与video文件名，命名image子文件夹并进行操作
            # 'xxx/video_data/ A4C/AAA.avi' --> 'xxx/image_data/ A4C/AAA.avi/*.jpg'
            subclass = split_filepath(subdir_v,1,1) # split_filepath('xxx/B/M/N/',1,3) = 'B/M/N'
            outdir_img = os.path.join(rootdir_i, subclass, name)
            video_path = os.path.join(subdir_v, file_name)
            # -------------------- 使用opencv-python将视频解析为图像 -------------------- #
            if not os.path.exists(outdir_img): os.makedirs(outdir_img)
            # 若目标文件夹不为空，且选择覆盖原有数据，则直接直接删除目标文件夹
            if len(os.listdir(outdir_img)) > 0:
                if OVERWRITE:
                    shutil.rmtree(outdir_img)
                    print('remove {}'.format(outdir_img))
                    os.makedirs(outdir_img)
                else:
                    continue
            # 打开视频文件，解析为图像并保存
            cap = cv2.VideoCapture(video_path)
            num = 1
            while True:
                # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
                success, data = cap.read()
                if not success:
                    break
                img = copy.deepcopy(data)
                img_path = '{}/image_{:05d}.jpg'.format(outdir_img, num)
                # cv2.imwrite() 第3个参数针对特定的格式：
                # 对于JPEG，表示图像质量（0~100，默认95）; 对于png，表示压缩级别（0~9，默认3）
                cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                num += 1
            cap.release()


def count_mask(img_gray,mask_gray_all,gap=5.0,return_num=3):
    # 求两直线交点
    def _get_line_cross_point(line1, line2):
        def _calc_abc_from_line_2d(x0, y0, x1, y1):
            a = y0 - y1
            b = x1 - x0
            c = x0 * y1 - x1 * y0
            return a, b, c
        # x1y1x2y2
        a0, b0, c0 = _calc_abc_from_line_2d(*line1)
        a1, b1, c1 = _calc_abc_from_line_2d(*line2)
        D = a0 * b1 - a1 * b0
        if D == 0:
            return None
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        # print(x, y)
        return x, y

    # 延长直线
    def _extend_line(x1_int, y1_int, x2_int, y2_int, x_int, y_int, basepix=8,flag=0):
        x1, y1, x2, y2 = float(x1_int),float(y1_int),float(x2_int),float(y2_int)
        x, y = int(x_int-1-basepix),int(y_int-1-basepix)
        if flag == 1:
            if y1 == y2:
                return basepix, y1, x, y2
            else:
                k = (y2 - y1) / (x2 - x1)
                b = (x1 * y2 - x2 * y1) / (x1 - x2)
                x3 = basepix
                y3 = b
                x4 = x
                y4 = int(k * x4 + b)
            return x3, y3, x4, y4
        else:
            if x1 == x2:
                return x1, basepix, x2, y
            else:
                k = (y2 - y1) / (x2 - x1)
                b = (x1 * y2 - x2 * y1) / (x1 - x2)
                y3 = basepix
                x3 = int(-1 * b / k)
                y4 = y
                x4 = int((y4 - b) / k)
                return x3, y3, x4, y4

    gray = copy.deepcopy(img_gray)
    mask_all = copy.deepcopy(mask_gray_all)
    gray[mask_all < 128] = 0
    edges = cv2.Canny(mask_all, 64, 224, L2gradient=True)
    # 统计概率霍夫线变换
    lines = cv2.HoughLinesP(edges, 0.8, np.pi / 180, 90, minLineLength=24, maxLineGap=16)
    #
    cross_pt = _get_line_cross_point(lines[0][0],lines[1][0])
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[return_num-2]
    max_contour = copy.deepcopy(contours[0])
    if len(contours)>1:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_contour = contours[np.argmax(area)][0]
    points = np.asarray(max_contour).transpose(0,2,1)
    cross_pts = np.expand_dims(np.tile(np.asarray(cross_pt),(len(max_contour),1)),-1)
    distances = np.round(np.linalg.norm(points-cross_pts, ord=2, axis=1)/float(gap)).squeeze().astype(np.int16)
    distance = np.argmax(np.bincount(distances))
    sector_radius = int(np.ceil(distance*float(gap)).item())

    temp_mask,real_mask = np.zeros_like(gray),np.zeros_like(gray)
    # 绘制圆圈
    rows, cols = gray.shape[0], gray.shape[1]
    cv2.circle(temp_mask, (int(cross_pt[0]),int(cross_pt[1])), sector_radius, (255, 255, 255),thickness=1)
    # cv2.circle(temp_mask, (int(cross_pt[0]), int(cross_pt[1])), sector_radius, (255, 0, 0), thickness=1)
    # 绘制检测+延长后的扇形边界直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x3, y3, x4, y4 = _extend_line(x1, y1, x2, y2,cols,rows, basepix=8,flag=0)
        x3, y3, x4, y4 = int(x3),int(y3),int(x4),int(y4)
        cv2.line(temp_mask, (x3, y3), (x4, y4), (255, 255, 255), thickness=1)
        # cv2.line(temp_mask, (x3, y3), (x4, y4), (0, 0, 255), thickness=1)
    cv2.rectangle(temp_mask, (0, 0), (cols, rows), (0, 0, 0), 32)
    contours = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[return_num-2]
    max_contour = copy.deepcopy(contours)
    if len(contours) > 1:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_contour = contours[np.argmax(area)]
    cv2.drawContours(real_mask, max_contour, -1, (255, 255, 255), cv2.FILLED)
    kernel_o = create_kernel(3)
    real_mask = cv2.morphologyEx(real_mask, cv2.MORPH_OPEN, kernel_o)

    return real_mask.astype(np.uint8)


'''将超声心动图(*.avi)解析为JPEG图像'''
def video2img_w_preprocessing(rootdir_video, rootdir_img):
    """
    :param rootdir_video: 视频文件输入根目录
    :param rootdir_img: 解析后图像输出根目录
    :return: 超声心动图(*.avi)解析得到的JPEG图像
    """
    # 检查输入路径是否存在，若存在则递归创建输出根目录
    rootdir_v = copy.deepcopy(rootdir_video)
    if not os.path.isdir(rootdir_v): return
    rootdir_i = copy.deepcopy(rootdir_img)
    if not os.path.exists(rootdir_i): os.makedirs(rootdir_i)
    # 获取“各子文件夹-子文件夹内文件”的字典
    filedict_video = get_filedirdict(rootdir_v, suffix='.avi')
    return_num = get_findContours_renum()

    for subdir_v,file_names in filedict_video.items():
        count = 0
        for file_name in file_names:
            count += 1
            if count < 2:
                continue
            # 若文件后缀不为*.avi则直接跳过
            if not file_name.endswith('.avi'):
                continue
            name, ext = os.path.splitext(file_name)
            # 以N种标准切面名称（对应video子文件夹名）与video文件名，命名image子文件夹并进行操作
            # 'xxx/video_data/ A4C/AAA.avi' --> 'xxx/image_data/ A4C/AAA.avi/*.jpg'
            subclass = split_filepath(subdir_v,1,1) # split_filepath('xxx/B/M/N/',1,3) = 'B/M/N'
            outdir_img = os.path.join(rootdir_i, subclass, name)
            video_path = os.path.join(subdir_v, file_name)
            # -------------------- 使用opencv-python将视频解析为图像 -------------------- #
            if not os.path.exists(outdir_img): os.makedirs(outdir_img)
            # 若目标文件夹不为空，且选择覆盖原有数据，则直接直接删除目标文件夹
            if len(os.listdir(outdir_img)) > 0:
                if OVERWRITE:
                    shutil.rmtree(outdir_img)
                    print('remove {}'.format(outdir_img))
                    os.makedirs(outdir_img)
                else:
                    continue
            # 打开视频文件，解析为图像
            cap = cv2.VideoCapture(video_path)
            img_list_bgr,img_list_gray = [],[]
            while True:
                # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
                success, data = cap.read()
                if not success:
                    break
                # opencv-python颜色通道顺序为BGR
                img_bgr = copy.deepcopy(data)
                img_list_bgr.append(img_bgr)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                img_list_gray.append(img_gray)
            cap.release()
            data_bgr = np.array(img_list_bgr)
            data_gray = np.array(img_list_gray)
            # 计算roi mask
            mask_all = np.zeros_like(data_gray)
            for i in range(data_gray.shape[0]):
                mask_gray = get_img_roi(copy.deepcopy(data_gray[i,:,:]),
                                        re_contours=False, pixel_th=2, k_sizes=[2, 5, 7])
                mask_all[i,:,:] = mask_gray
            mask_gray_all = np.sum(copy.deepcopy(mask_all).astype(np.float64), axis=0)
            mask_gray_all = (np.clip(mask_gray_all, 0.0, 1.0)*255.0).astype(np.uint8)

            img_gray = copy.deepcopy(data_gray[0,...])
            # 若conut_mask为空，则使用上一视频的mask
            img_mask = count_mask(img_gray, mask_gray_all, return_num=return_num)
            first_row,last_row,first_col,last_col = get_img_range(
                img_mask, retype='none', ismask=True, pixel_th=25, k_sizes=[1,1,1])

            data_mask = np.tile(np.expand_dims(img_mask,-1), (1, 1, 3))
            data_mask = np.tile(np.expand_dims(data_mask,0), (data_gray.shape[0], 1, 1, 1))
            data_out = copy.deepcopy(data_bgr)
            data_out[data_mask<128] = 0
            data_out = data_out[:,first_row:last_row+1,first_col:last_col+1,:]

            # for i in range(data_out.shape[0]):
            #     img_w = copy.deepcopy(data_out[i,...])
            #     # cv2.imwrite() 第3个参数针对特定的格式：
            #     # 对于JPEG，表示图像质量（0~100，默认95）; 对于png，表示压缩级别（0~9，默认3）
            #     img_path = '{}/image_{:05d}.jpg'.format(outdir_img, i+1)
            #     cv2.imwrite(img_path, img_w, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imshow('img_bgr', data_bgr[0,...])
            cv2.imshow('img_out', data_out[0,...])
            cv2.imshow('img_mask', img_mask)
            cv2.waitKey(10)
            cv2.waitKey(0)



'''生成视频frame总数文件n_frames'''
def gen_nframes(rootdir_img, cover=True):
    """
    :param rootdir_img: 解析后图像输入根目录
    :param cover:       是否覆盖原有frame总数文件n_frames
    :return:            视频frame总数文件n_frames
    """
    rootdir = copy.deepcopy(rootdir_img)
    if not os.path.isdir(rootdir): return
    # 获取“各子文件夹-子文件夹内文件”的字典
    filedict_image = get_filedirdict(rootdir, suffix='.jpg')

    for subdir_name, file_names in filedict_image.items():
        if cover or (not os.path.exists(os.path.join(subdir_name, 'n_frames'))):
            img_indices = []
            for img_filename in file_names:
                if not ('image' in img_filename or img_filename.endswith('.jpg')):
                    continue
                # img_filename = 'image_00001.jpg'
                img_idx_num = int(img_filename.split('.')[0].split('_')[-1])
                img_indices.append(img_idx_num)

            if len(img_indices) > 0:
                if len(img_indices) != (max(img_indices)-min(img_indices)+1):
                    print("len(img_indices) != max(img_indices), please check the images!")

            n_frames = len(img_indices) if len(img_indices) > 0 else 0
            subdir_temp = split_filepath(subdir_name,1,2)
            print_str = '{}    n_frames:{}'.format(subdir_temp, n_frames) \
                if n_frames > 0 else '{} no image files'.format(subdir_temp)
            print(print_str)
            # 文件夹内写入frame总数
            with open(os.path.join(subdir_name, 'n_frames'), 'w') as dst_file:
                dst_file.write(str(n_frames))




if __name__=="__main__":
    rootdir_video = ROOTDIR_VIDEO
    rootdir_img = ROOTDIR_IMAGE

    # video2img(rootdir_video=rootdir_video, rootdir_img=rootdir_img)
    video2img_w_preprocessing(rootdir_video=rootdir_video, rootdir_img=rootdir_img)
    # gen_nframes(rootdir_img=rootdir_img, cover=True)


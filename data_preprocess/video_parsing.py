#! /usr/bin/env python
# -*- coding=utf-8 -*-
# 将超声心动图(*.avi)解析为JPEG图像；并生成n_frames
# Author: SEU-BME-LBMD-zzy,SEU-BME-LBMD-chl,SEU-BME-LBMD-cyc
import os
import cv2
import copy
import math
import shutil
import itertools
import numpy as np
from extern.fileopts import split_filename,split_filepath,get_filedirdict
from extern.get_roi_and_datarange import get_img_roi,get_img_range,create_kernel,get_findContours_renum


# Global Constants
ROOTDIR = 'F:/BME_New/Echocardiography_Datas/Quality_Assessment_object_detection/'
# ROOTDIR_VIDEO = ROOTDIR + 'video_data/A4C/'     # 视频文件输入根目录
# ROOTDIR_IMAGE = ROOTDIR + 'image_data_processed/'         # 解析图像输出根目录
ROOTDIR_VIDEO = ROOTDIR + 'video_data/PSAX_GV_ext_test2/'     # 视频文件输入根目录
ROOTDIR_IMAGE = ROOTDIR + 'video_data_processed/'         # 解析图像输出根目录
OVERWRITE = True                                # 是否覆盖原有数据
# OVERWRITE = False                                # 是否覆盖原有数据



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


'''将超声心动图(*.avi)解析为JPEG图像'''
def video2img_w_preprocessing(rootdir_video, rootdir_img, original_mask_path, outflag='img'):
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
        count_min = 0
        last_mask = cv2.imread(original_mask_path, 0)
        for file_name in file_names:
            count += 1
            if count < count_min:
                continue
            # 若文件后缀不为*.avi则直接跳过
            if not file_name.endswith('.avi'):
                continue
            name, ext = os.path.splitext(file_name)
            # 以N种标准切面名称（对应video子文件夹名）与video文件名，命名image子文件夹并进行操作
            # 'xxx/video_data/ A4C/AAA.avi' --> 'xxx/image_data/ A4C/AAA.avi/*.jpg'
            subclass = split_filepath(subdir_v,1,1) # split_filepath('xxx/B/M/N/',1,3) = 'B/M/N'
            video_path = os.path.join(subdir_v, file_name)
            outdir_img = os.path.join(rootdir_i, subclass, name)
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
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
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
            # 计算roi mask并处理数据，完成后写入图像
            img_mask = proprecess_w_mask(
                file_name,data_bgr,data_gray,last_mask,outdir_img,
                outflag=outflag,frame_rate=frame_rate,return_num=return_num)
            if isinstance(img_mask, np.ndarray):
                last_mask = copy.deepcopy(img_mask)
                cv2.imwrite(original_mask_path, last_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


'''将超声心动图(*.avi)解析+处理输出为视频'''
def video2video_w_preprocessing(rootdir_video, rootdir_img, original_mask_path, outflag='video'):
    """
    :param rootdir_video: 视频文件输入根目录
    :param rootdir_img: 解析后视频输出根目录
    :return: 超声心动图(*.avi)解析+处理后的视频
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
        count_min = 0
        last_mask = cv2.imread(original_mask_path, 0)

        # 以N种标准切面名称（对应video子文件夹名）与video文件名，命名image子文件夹并进行操作
        # 'xxx/video_data/ A4C/AAA.avi' --> 'xxx/image_data/ A4C/AAA.avi/*.jpg'
        subclass = split_filepath(subdir_v, 1, 1)  # split_filepath('xxx/B/M/N/',1,3) = 'B/M/N'
        outdir_img = os.path.join(rootdir_i, subclass)
        if not os.path.exists(outdir_img): os.makedirs(outdir_img)
        # 若目标文件夹不为空，且选择覆盖原有数据，则直接直接删除目标文件夹
        if len(os.listdir(outdir_img)) > 0:
            if OVERWRITE:
                shutil.rmtree(outdir_img)
                print('remove {}'.format(outdir_img))
                os.makedirs(outdir_img)
            else:
                continue

        for file_name in file_names:
            count += 1
            if count < count_min:
                continue
            # 若文件后缀不为*.avi则直接跳过
            if not file_name.endswith('.avi'):
                continue
            video_path = os.path.join(subdir_v, file_name)
            # -------------------- 使用opencv-python将视频解析为图像 -------------------- #
            # 打开视频文件，解析为图像
            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
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
            # 计算roi mask并处理数据，完成后写入图像
            img_mask = proprecess_w_mask(
                file_name,data_bgr,data_gray,last_mask,outdir_img,
                outflag=outflag,frame_rate=frame_rate,return_num=return_num)
            if isinstance(img_mask, np.ndarray):
                last_mask = copy.deepcopy(img_mask)
                cv2.imwrite(original_mask_path, last_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


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

###############################################################################
# Helper Functions
###############################################################################
'''根据mask处理数据'''
def proprecess_w_mask(file_name,data_bgr,data_gray,last_mask,outdir_img,outflag='img',frame_rate=1.0,return_num=3):
    """
    :param file_name:     处理的视频文件的文件名（用于参数计算异常时的提示）
    :param data_bgr:      BGR格式的3D视频数据（opencv为rgb格式)
    :param data_gray:     Gray格式的3D视频数据
    :param last_mask:     上一个超声视频的扇形区域mask(仅在当前视频参数异常时起作用）
    :param outdir_img:    处理后图像输出子文件夹路径
    :param return_num:    cv2.findContours()函数返回的参数个数，与opencv-python库版本有关
    :return new_mask:     更新后的last_mask
    """
    # 各帧形态学操作得到mask_frame，并在帧维度上投影叠加，获取整个视频的mask_all
    mask_frames = np.zeros_like(data_gray)
    # 考虑到图像中辅助性信息位置并不固定，因此并未采用直接全部投影到Z轴再形态学操作的方式
    # 后续优化可基于numpy.split()函数，但切分参数indices_or_sections不宜过大
    for i in range(data_gray.shape[0]):
        mask_frame = get_img_roi(copy.deepcopy(data_gray[i, :, :]),
                                re_contours=False, pixel_th=2, k_sizes=[1, 2, 3])
        mask_frames[i,:,:] = mask_frame
    mask_all = np.sum(copy.deepcopy(mask_frames).astype(np.float64), axis=0)
    mask_all = (np.clip(mask_all, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_all = get_img_roi(copy.deepcopy(mask_all),re_contours=False, pixel_th=128, k_sizes=[1, 3, 7])
    # 计算整个视频的mask
    img_mask = count_mask(file_name,mask_all, return_num=return_num)
    # 若conut_mask为空，则使用上一视频的mask；否则更新mask
    new_mask = 0
    if isinstance(img_mask, np.ndarray):
        new_mask = copy.deepcopy(img_mask)
    else:
        # 数据尺寸有可能不一致，需要检查是否resize
        replace_mask = copy.deepcopy(last_mask)
        if not (replace_mask.shape==mask_all.shape):
            # cv2.resize()在dsize形参中写(rows,cols)，得到的是(cols,rows)的结果
            replace_mask = cv2.resize(
                last_mask, (mask_all.shape[1],mask_all.shape[0]), interpolation=cv2.INTER_LINEAR)
            replace_mask[np.where(replace_mask<64)] = 0
        img_mask = copy.deepcopy(replace_mask)
    # 根据mask裁剪图像
    first_row, last_row, first_col, last_col = get_img_range(
        img_mask, retype='none', ismask=True, pixel_th=25, k_sizes=[1, 1, 1])
    data_mask = np.tile(np.expand_dims(img_mask, -1), (1, 1, 3))
    data_mask = np.tile(np.expand_dims(data_mask, 0), (data_gray.shape[0], 1, 1, 1))
    data_out = copy.deepcopy(data_bgr)
    data_out[data_mask < 128] = 0
    data_out = data_out[:, first_row:last_row+1, first_col:last_col+1, :]

    # 将视频数据写入JPEG图像或视频
    write_imgorvideo(data_out=data_out,outdir=outdir_img,file_name=file_name,outflag=outflag,frame_rate=frame_rate)

    return new_mask


'''计算mask'''
def count_mask(file_name,mask_gray_all,gap=8.0,return_num=3):
    """
    :param file_name:     处理的视频文件的文件名（用于参数计算异常时的提示）
    :param mask_gray_all: 整个视频的mask（灰度图）
    :param gap:           扇形半径计算时，轮廓点到圆心距离的量化尺度（单位：像素）
    :param return_num:    cv2.findContours()函数返回的参数个数，与opencv-python库版本有关
    :return:              根据所求扇形几何参数划定扇形范围mask
    """
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
    def _extend_line(line, x_int, y_int, basepix=8,flag=0):
        x1, y1, x2, y2 = float(line[0]),float(line[1]),float(line[2]),float(line[3])
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

    # -------------------------------- 修正传入的mask -------------------------------- #
    mask_all_temp = copy.deepcopy(mask_gray_all)
    # 计算最大连通集
    contours = cv2.findContours(
        mask_all_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[return_num-2]
    img_temp = np.zeros(mask_gray_all.shape, np.uint8)
    cv2.drawContours(img_temp, contours, -1, (255, 255, 255), cv2.FILLED)
    # 获取最大连通集的最大轮廓，以该轮廓作为扇形区域轮廓
    contours = cv2.findContours(
        img_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[return_num-2]
    max_contour = copy.deepcopy(contours)
    if len(contours) > 1:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_contour = tuple(np.expand_dims(contours[np.argmax(area)], 0))
    img_temp = np.zeros(mask_gray_all.shape, np.uint8)
    cv2.drawContours(img_temp, max_contour, -1, (255, 255, 255), cv2.FILLED)
    morph_kernel = create_kernel(2)
    mask_all = cv2.morphologyEx(cv2.morphologyEx(
        img_temp, cv2.MORPH_OPEN, morph_kernel), cv2.MORPH_CLOSE, morph_kernel)

    # -------------- 计算扇形径线参数、圆心与半径参数（所得参数异常时直接返回0） -------------- #
    rows, cols = mask_all.shape[0], mask_all.shape[1]
    # Canny算子计算边缘
    edges = cv2.Canny(mask_all, 64, 224, L2gradient=True)
    # 统计概率霍夫线变换，得到扇形的两条径线所在的直线
    # lines_original = cv2.HoughLinesP(edges, 0.8, np.pi/180, threshold=64, minLineLength=16, maxLineGap=32)
    # theta=[0.5236, 2.618]，霍夫变换角度单位为弧度制，对应角度(theta*180/pi)=[30°, 150°]
    hough_lines = cv2.HoughLines(edges, 1.0, np.pi/180, threshold=64, min_theta=0.5,max_theta=2.7)
    if hough_lines is None:
        cv2.imshow('abnormal_edges', edges)
        cv2.waitKey(5)
        return 0
    # 筛除斜率绝对值过低、以及斜率与已存在直线过于接近的直线
    lines = []
    slopes = []
    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = float(x0 + cols * (-b))
        y1 = float(y0 + cols * (a))
        x2 = float(x0 - cols * (-b))
        y2 = float(y0 - cols * (a))
        # temp_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # cv2.line(temp_mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=3)
        # cv2.namedWindow('abnormal_curve_detection', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("abnormal_curve_detection", int(cols / 2.0), int(rows / 2.0))
        # cv2.imshow('abnormal_curve_detection', temp_mask)
        # cv2.waitKey(5)
        slope = (y2-y1)/(np.abs(x2-x1)+1e-5)
        if (0.25 <= abs(slope) and abs(slope) <= 10.0):
            add_flag = True
            for slope_i in slopes:
                if np.abs(slope-slope_i)<0.1:
                    add_flag = False
                    break
            if add_flag:
                newline = [int(x1),int(y1),int(x2),int(y2)]
                lines.append(newline)
                if slope not in slopes:
                    slopes.append(slope)
    lines = np.asarray(lines)

    if not (len(lines)==2 or len(lines)==4):
        print("please check the params of cv2.HoughLines(),"
              " as len(lines) != 2or4 in file {} !".format(file_name))
        # 显示扇形径线参数计算异常的样本
        temp_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for line in lines:
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
            cv2.line(temp_mask, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
        cv2.namedWindow('abnormal_curve_detection',cv2.WINDOW_NORMAL)
        cv2.resizeWindow("abnormal_curve_detection", int(cols/2.0), int(rows/2.0))
        cv2.imshow('abnormal_curve_detection', temp_mask)
        cv2.waitKey(5)
        return 0
    # 根据得到的两条径线所在的直线，计算径线交点（扇形圆心）
    cross_pt = _get_line_cross_point(lines[0],lines[1])
    if not ((0<=cross_pt[0] and cross_pt[0]<cols) and (0<=cross_pt[1] and cross_pt[1]<rows)):
        print("0<= cross_pt <shape!")
        return 0
    # 根据轮廓上各点 到 两径线交点（圆心）的距离，决定扇形半径
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[return_num-2]
    max_contour = copy.deepcopy(contours[0])
    if len(contours)>1:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_contour = contours[np.argmax(area)][0]
    edge_points = np.asarray(max_contour).transpose(0,2,1)
    cross_pts = np.expand_dims(np.tile(np.asarray(cross_pt),(len(max_contour),1)),-1)
    distances = np.round(np.linalg.norm(edge_points-cross_pts, ord=2, axis=1)/float(gap)).squeeze().astype(np.int16)
    distance = np.argmax(np.bincount(distances))
    sector_radius = int(np.ceil(distance*float(gap)).item())
    if not (0.5*rows<=sector_radius and sector_radius<=cols and sector_radius<=rows):
        print("sector_radius might be worng!")
        return 0

    # ------------------------ 根据扇形径线参数、圆心与半径参数划定扇形范围 ------------------------ #
    temp_mask,real_mask = np.zeros_like(mask_all),np.zeros_like(mask_all)
    # 绘制扇形弧线
    cv2.circle(temp_mask, (int(cross_pt[0]),int(cross_pt[1])), sector_radius, (255, 255, 255),thickness=1)
    # 绘制检测+延长后的扇形径线
    for line in lines:
        line_pts = [pt.item() for pt in line]
        x3, y3, x4, y4 = _extend_line(line_pts,cols,rows, basepix=8,flag=0)
        x3, y3, x4, y4 = int(x3),int(y3),int(x4),int(y4)
        cv2.line(temp_mask, (x3, y3), (x4, y4), (255, 255, 255), thickness=1)
    cv2.rectangle(temp_mask, (0, 0), (cols, rows), (0, 0, 0), 32)
    # 以扇形轮廓的填充作为扇形区域mask
    contours = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[return_num-2]
    max_contour = copy.deepcopy(contours)
    if len(contours) > 1:
        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_contour = tuple(np.expand_dims(contours[np.argmax(area)],0))
    cv2.drawContours(real_mask, max_contour, -1, (255, 255, 255), cv2.FILLED)
    kernel_o = create_kernel(2)
    real_mask = cv2.morphologyEx(real_mask, cv2.MORPH_OPEN, kernel_o).astype(np.uint8)

    return real_mask


'''写入图像或视频'''
def write_imgorvideo(data_out,outdir,file_name,outflag='img',frame_rate=1.0):
    if outflag=='img':
        outdir_img = copy.deepcopy(outdir)
        for i in range(data_out.shape[0]):
            img_w = copy.deepcopy(data_out[i, ...])
            # cv2.imwrite() 第3个参数针对特定的格式：
            # 对于JPEG，表示图像质量（0~100，默认95）; 对于png，表示压缩级别（0~9，默认3）
            img_path = '{}/image_{:05d}.jpg'.format(outdir_img, i + 1)
            cv2.imwrite(img_path, img_w, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif outflag=='video':
        # 输出为视频
        video_path = os.path.join(outdir,file_name)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(video_path, fourcc, frame_rate, (data_out.shape[-2], data_out.shape[-3]))
        for i in range(data_out.shape[0]):
            frame = copy.deepcopy(data_out[i, ...])
            # write the flipped frame
            out.write(frame)
        # Release everything if job is finished
        out.release()


if __name__=="__main__":
    rootdir_video = ROOTDIR_VIDEO
    rootdir_img = ROOTDIR_IMAGE
    original_mask_path = './extern/original_mask.jpg'

    # video2img(rootdir_video=rootdir_video, rootdir_img=rootdir_img)
    video2img_w_preprocessing(
        rootdir_video=rootdir_video, rootdir_img=rootdir_img, original_mask_path=original_mask_path, outflag='img')
    # video2video_w_preprocessing(
    #     rootdir_video=rootdir_video, rootdir_img=rootdir_img, original_mask_path=original_mask_path, outflag='video')
    # gen_nframes(rootdir_img=rootdir_img, cover=True)


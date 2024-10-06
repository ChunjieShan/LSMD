#! /usr/bin/env python
# -*- coding=utf-8 -*-
# Get the range of valid data in the 3D numpy array
# 获取3D numpy数组中有效数据的范围
# -------------------------------------------------------------
# get_img_roi() 获取2D图像的roi mask
# get_img_range() 获取2D图像非空数据范围(行列)
# get_npy_range() 获取3D图像非空数据范围(行列)
# get_slices_range() 获取3D numpy数组非空数据范围（层数）
# -------------------------------------------------------------
# Author: SEU-BME-LBMD-chl,SEU-BME-LBMD-zzy
import cv2
import copy
import math
import numpy as np


# Global Constants
DATA_MIN=-1000
HU_RANGE=3000.0
HU_OFFSET=-1000.0
K_SIZES=[2,5,15]



'''获取2D图像的roi mask'''
def get_img_roi(img_r,re_contours=False,pixel_th=25,k_o=None,k_c=None,k_e=None,k_sizes=K_SIZES):
    """
    :param img_r: 读入的np.uint8图像
    :param re_contours: 是否返回相应的轮廓
    :param pixel_th: 有效数据阈值
    :param k_o: 初始开闭操作卷积核
    :param k_c: 进一步开闭操作卷积核
    :param k_e: 最终开闭操作卷积核
    :param k_sizes: 初始、进一步、最终开闭操作卷积核尺寸
    :return: np.uint8类型的roi mask以及轮廓
    """
    pixel_m = 255  #像素最大值
    size_o, size_c, size_e = 0, 0, 0
    return_num = get_findContours_renum()  #用于opencv-python不同版本处理
    if (k_o is None) or (k_c is None) or (k_e is None):
        size_o, size_e = k_sizes[0],k_sizes[-1]
        size_c = k_sizes[1] if len(k_sizes)==3 else int((size_o+size_e)/3.0)
    kernel_o = create_kernel(size_o) if k_o is None else k_o
    kernel_c = create_kernel(size_c) if k_c is None else k_c
    kernel_e = create_kernel(size_e) if k_e is None else k_e
    img_copy = copy.deepcopy(img_r)
    assert isinstance(img_r[0,0],np.uint8),"type of img_r should be np.uint8!!!"
    # 阈值分割->开运算->闭运算
    img_w1 = cv2.threshold(img_copy, pixel_th, pixel_m, cv2.THRESH_BINARY)[-1] #返回thresh,img_w
    closing1 = cv2.morphologyEx(cv2.morphologyEx(
        img_w1, cv2.MORPH_OPEN, kernel_o), cv2.MORPH_CLOSE, kernel_o)
    closing2 = cv2.morphologyEx(cv2.morphologyEx(
        closing1, cv2.MORPH_OPEN, kernel_c), cv2.MORPH_CLOSE, kernel_c)
    # 寻找闭运算后的轮廓
    # opencv-python 2.x与不低于3.5版本两个返回contours,hierarchy，而3.0~3.4.x版本还返回image
    contours1 = cv2.findContours(
        copy.deepcopy(closing2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[return_num-2]
    # 以255填充轮廓内所有像素点
    img_temp2 = np.zeros(img_r.shape, np.uint8)
    cv2.drawContours(img_temp2,contours1,-1,(pixel_m,pixel_m,pixel_m),cv2.FILLED)
    # 大卷积核以消除连通集的弱粘连
    opening3 = cv2.morphologyEx(img_temp2, cv2.MORPH_OPEN, kernel_e)
    # 寻找消除过连通集合弱粘连后的图像轮廓
    # opencv-python 2.x与不低于3.5版本两个返回contours,hierarchy，而3.0~3.4.x版本返回img,countours,hierarchy
    contours2 = cv2.findContours(
        copy.deepcopy(opening3), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[return_num-2]
    # 获取最大轮廓
    area = []
    for k in range(len(contours2)):
        area.append(cv2.contourArea(contours2[k]))
    # ---------------------- 处理躯干部分大于ROI的情况 ---------------------- #
    img_w3 = np.zeros(img_r.shape, np.uint8)
    if len(area) > 1:
        max_idx = np.argmax(area)
        # 去除除最大轮廓外长宽比例不对的轮廓
        contourslist = []
        for k in range(len(contours2)):
            rect = cv2.minAreaRect(contours2[k])
            aspect_ratio = rect[1][0] / rect[1][1]
            add_flag = True
            if (aspect_ratio < 0.35) or (aspect_ratio > 1.0 / 0.35):
                if k != max_idx:
                    add_flag = False
            if add_flag:
                contourslist.append(copy.deepcopy(contours2[k]))
        contours2 = tuple(contourslist)
        # ROI模板
        cv2.drawContours(img_w3, contours2, -1, (pixel_m, pixel_m, pixel_m), cv2.FILLED)
    elif np.sum(img_w1,axis=(0,1)) > img_w1.shape[0]*img_w1.shape[1]*pixel_m/64:
        img_w3 = np.ones(img_r.shape, np.uint8)*pixel_m

    # cv2.imshow('img_w',img_w3)
    # cv2.waitKey(100)

    if not re_contours:
        return img_w3
    else:
        return img_w3, contours2


'''获取2D图像非空数据范围'''
def get_img_range(img_r,retype='list',ismask=False,pixel_th=25,
    k_sizes=K_SIZES,data_min=DATA_MIN,hu_range=HU_RANGE,hu_offset=HU_OFFSET):
    """
    :param img_r: 读入的2D图像
    :param retype: 返回的数据范围格式（是否合并为列表）
    :param ismask: 读入的2D np.uint8图像本身是否为roi mask
    :param pixel_th: 有效数据阈值
    :param k_sizes: 初始、进一步、最终开闭操作卷积核尺寸
    :return: 2D图像非空数据范围(行列)
    """
    pixel_m = 255  #像素最大值
    part_num = 5   #最值区间等分数目（256/part_num）
    pixel_d,pixel_u = int(pixel_m/part_num),pixel_m-int(pixel_m/part_num)
    assert (retype=='list' or retype=='none'),"retype should be list or none!!!"
    # np.uint8与np.uint16统一处理为np.uint8（与后续调用get_img_range不冲突）
    img_data = copy.deepcopy(img_r)
    if isinstance(img_data[0,0],np.int16):
        npyimg = copy.deepcopy(img_data).astype(np.float)
        data_min = hu_offset if data_min < 0 else 0
        img_data = np.uint8((npyimg-data_min)/hu_range*255.0)
    elif isinstance(img_data[0,0],np.uint8):
        img_data = copy.deepcopy(img_r)
    else:
        raise ValueError("data type of img_r should be np.int16 or np.uint8!!!")
    # 较小值与大值均设置为0，判断是否为mask,若不为mask,求roi mask
    img_temp = copy.deepcopy(img_data)
    img_temp[img_data<pixel_d] = 0
    img_temp[img_data>pixel_u] = 0
    if (np.sum(img_temp) < 256*pixel_m) and (not ismask):
        img_data = get_img_roi(img_data,pixel_th=pixel_th,k_sizes=k_sizes)

    rows, cols = img_data.shape[0], img_data.shape[1]
    first_row,last_row,first_col,last_col = 0,rows-1,0,cols-1
    # first row
    for row in range(rows):
        if np.sum(img_data[row,:]) > 8:
            first_row = row
            break
    # last row
    for row in range(rows):
        if np.sum(img_data[rows-1-row,:]) > 8:
            last_row = rows-1-row
            break
    # first col
    for col in range(cols):
        if np.sum(img_data[:,col]) > 8:
            first_col = col
            break
    # last row
    for col in range(cols):
        if np.sum(img_data[:,cols-1-col]) > 8:
            last_col = cols-1-col
            break

    if retype=='list':
        return [first_row,last_row,first_col,last_col]
    elif retype=='none':
        return first_row,last_row,first_col,last_col
    else:
        raise ValueError("retype should be list or none!!!")


'''获取中心截取2D图像的坐标（左上角）'''
def crop_img_range(img_r,crop_size,retype='none',pixel_th=25,
    k_sizes=K_SIZES,data_min=DATA_MIN,hu_range=HU_RANGE,hu_offset=HU_OFFSET):
    """
    :param img_r: 读入的2D图像(np.uint8型或np.uint16型)
    :param crop_size: 截取的数据范围
    :param retype: 返回的数据范围格式（是否合并为列表）
    :param pixel_th: 有效数据阈值
    :param k_sizes: 初始、进一步、最终开闭操作卷积核尺寸
    :param data_min: 2D图像的最小值
    :param hu_range,hu_offset: Hu值的取值范围以及最小值
    :return: 围绕非空数据范围中心截取2D图像的左上角坐标
    """
    img_data = copy.deepcopy(img_r)
    rows, cols = img_data.shape[-2], img_data.shape[-1]
    first_row,last_row,first_col,last_col = \
        get_img_range(img_data,
                      retype='none',
                      pixel_th=pixel_th,
                      k_sizes=k_sizes,
                      data_min=data_min,
                      hu_range=hu_range,
                      hu_offset=hu_offset)
    # 倾向于向左+向下数据范围
    col_left,row_up = 0,0
    if first_col+last_col-crop_size<0:
        col_left = 0
    elif first_col+last_col+crop_size>=2*(cols-1):
        col_left = cols-1-crop_size
    else:
        col_left = int((first_col+last_col-crop_size)/2.0)
    if first_row+last_row-crop_size<0:
        row_up = 0
    elif first_row+last_row+crop_size>=2*(rows-1):
        row_up = rows-1-crop_size
    else:
        row_up = math.ceil((first_row+last_row-crop_size)/2.0)

    if retype=='list':
        return [row_up,col_left]
    elif retype=='none':
        return row_up,col_left
    else:
        raise ValueError("retype should be list or none!!!")



'''获取3D图像非空数据范围'''
def get_npy_range(npyimg,retype='list',ismask=False,pixel_th=25,
    k_sizes=K_SIZES,data_min=DATA_MIN,hu_range=HU_RANGE,hu_offset=HU_OFFSET):
    """
    :param npyimg: 读入的3D numpy数组
    :param retype: 返回的数据范围格式（是否合并为列表）
    :param ismask: 读入的3D np.uint8图像本身是否为roi mask
    :param pixel_th: 有效数据阈值
    :param k_sizes: 初始、进一步、最终开闭操作卷积核尺寸
    :param data_min: 3D numpy数组的最小值
    :param hu_range,hu_offset: Hu值的取值范围以及最小值
    :return: 3D图像非空数据范围(行列)
    """
    f_l_r_c = [[], [], [], []]
    # np.uint8与np.uint16统一处理为np.uint8（与后续调用get_img_range不冲突）
    npydata = copy.deepcopy(npyimg)
    if isinstance(npyimg[0,0,0],np.int16):
        npydata = copy.deepcopy(npyimg).astype(np.float)
        data_min = hu_offset if data_min < 0 else 0
        npydata = np.uint8((npydata-data_min)/hu_range*255.0)
    elif isinstance(npyimg[0,0,0],np.uint8):
        npydata = copy.deepcopy(npyimg)
    else:
        raise ValueError("data type of npydata should be np.int16 or np.uint8!!!")
    # 获取各切片的非空数据范围(行列)
    for i in range(npyimg.shape[0]):
        img_data = npydata[i, :, :]
        first_row, last_row, first_col, last_col = \
            get_img_range(img_data,retype=retype,ismask=ismask,pixel_th=pixel_th,k_sizes=k_sizes)
        f_l_r_c[0].append(first_row)
        f_l_r_c[1].append(last_row)
        f_l_r_c[2].append(first_col)
        f_l_r_c[3].append(last_col)
    # 取各层非空数据范围(行列)极值作为整个3D数据体的数据范围(行列)
    first_row,last_row = min(f_l_r_c[0]),max(f_l_r_c[1])
    first_col,last_col = min(f_l_r_c[2]),max(f_l_r_c[3])

    if retype=='list':
        return [first_row,last_row,first_col,last_col]
    elif retype=='none':
        return first_row,last_row,first_col,last_col
    else:
        raise ValueError("retype should be list or none!!!")


'''获取中心截取3D图像的坐标（左上角）'''
def crop_npy_range(npyimg,crop_size,retype='none',pixel_th=25,
    k_sizes=K_SIZES,data_min=DATA_MIN,hu_range=HU_RANGE,hu_offset=HU_OFFSET):
    """
    :param npyimg: 读入的3D numpy数组(np.uint8型或np.uint16型)
    :param crop_size: 截取的数据范围（不涉及Z轴方向）
    :param retype: 返回的数据范围格式（是否合并为列表）
    :param pixel_th: 有效数据阈值
    :param k_sizes: 初始、进一步、最终开闭操作卷积核尺寸
    :param data_min: 3D numpy数组的最小值
    :param hu_range,hu_offset: Hu值的取值范围以及最小值
    :return: 围绕非空数据范围中心截取3D图像的左上角坐标
    """
    img_data = copy.deepcopy(npyimg)
    rows, cols = img_data.shape[-2], img_data.shape[-1]
    first_row,last_row,first_col,last_col = \
        get_npy_range(img_data,
                      retype='none',
                      pixel_th=pixel_th,
                      k_sizes=k_sizes,
                      data_min=data_min,
                      hu_range=hu_range,
                      hu_offset=hu_offset)
    # 倾向于向左+向下数据范围
    col_left,row_up = 0,0
    if first_col+last_col-crop_size<0:
        col_left = 0
    elif first_col+last_col+crop_size>=2*(cols-1):
        col_left = cols-1-crop_size
    else:
        col_left = int((first_col+last_col-crop_size)/2.0)
    if first_row+last_row-crop_size<0:
        row_up = 0
    elif first_row+last_row+crop_size>=2*(rows-1):
        row_up = rows-1-crop_size
    else:
        row_up = math.ceil((first_row+last_row-crop_size)/2.0)

    if retype=='list':
        return [row_up,col_left]
    elif retype=='none':
        return row_up,col_left
    else:
        raise ValueError("retype should be list or none!!!")



'''获取3D numpy数组非空数据范围（slice索引）'''
def get_slices_range(npydata,retype='list',pixel_th=25,
    k_sizes=K_SIZES,data_min=DATA_MIN,hu_range=HU_RANGE,hu_offset=HU_OFFSET):
    """
    :param npydata: 输入的3D numpy数组
    :param retype: 返回的数据范围格式（是否合并为列表）
    :param pixel_th: 有效数据阈值
    :param k_sizes: 初始、进一步、最终开闭操作卷积核尺寸
    :param data_min: 3D numpy数组的最小值
    :param hu_range,hu_offset: Hu值的取值范围以及最小值
    :return: 数组有效的数据范围（Z轴方向非空数据坐标的最小值与最大值）
    """
    v_m=255
    range_temp = []
    npyimg = copy.deepcopy(npydata)
    data_range_min, data_range_max = 0, npydata.shape[0]-1
    assert (retype=='list' or retype=='none'),"retype should be list or none!!!"
    if isinstance(npydata[0,0,0],np.int16):
        npyimg = copy.deepcopy(npydata).astype(np.float)
        data_min = hu_offset if data_min < 0 else 0
        npyimg = np.uint8((npyimg-data_min)/hu_range*255.0)
    elif isinstance(npydata[0,0,0],np.uint8):
        npyimg = copy.deepcopy(npydata)
    else:
        raise ValueError("data type of npydata should be np.int16 or np.uint8!!!")
    kernel_o = create_kernel(k_sizes[0])
    kernel_c = create_kernel(k_sizes[1])
    kernel_e = create_kernel(k_sizes[-1])
    pixel_th_erode = min(75,pixel_th*3)
    kernel_erode = create_kernel(int((k_sizes[1]+k_sizes[-1])/2.0))
    # 当经过系列操作后最大轮廓不为空时，认为数据非空
    for i in range(npyimg.shape[0]):
        img_r = copy.deepcopy(npyimg[i,:,:])
        img_w, contours = get_img_roi(
            img_r,re_contours=True,pixel_th=pixel_th,k_o=kernel_o,k_c=kernel_c,k_e=kernel_e,k_sizes=k_sizes)
        # 最大轮廓不为空时，认为数据非空
        if len(contours) > 0:
            thresh, img_temp = cv2.threshold(img_r, 128, v_m, cv2.THRESH_BINARY)
            if np.sum(img_temp) > 64*v_m:
                range_temp.append(i)
    # 再次筛选两端
    for i in range(len(range_temp)):
        img_r = copy.deepcopy(npyimg[range_temp[i],:,:])
        thresh, img_temp = cv2.threshold(img_r, pixel_th_erode, v_m, cv2.THRESH_BINARY)
        eroded = cv2.erode(img_temp,kernel_erode)
        if np.sum(eroded) > 256*v_m:
            data_range_min = range_temp[i]
            break
    for i in range(len(range_temp)):
        data_i = len(range_temp)-1-i
        img_r = copy.deepcopy(npyimg[data_i,:,:])
        thresh, img_temp = cv2.threshold(img_r, pixel_th_erode, v_m, cv2.THRESH_BINARY)
        eroded = cv2.erode(img_temp,kernel_erode)
        if np.sum(eroded) > 256*v_m:
            data_range_max = range_temp[data_i]
            break

    if retype=='list':
        return [data_range_min, data_range_max]
    elif retype=='none':
        return data_range_min, data_range_max


# ================================================================================ #
# ------------------- basic functions used only in this script ------------------- #

# 生成圆形卷积核
def create_kernel(kernel_r):
    """
    :param kernel_r: 指定的卷积核半径(d=2*r+1)
    :return: 距圆心kernel_r内为1，其余为0的kernel（np.uint8）
    """
    kernel_d = kernel_r * 2 + 1
    kernel = np.ones((kernel_d,kernel_d), np.uint8) * 255
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            if ((y-kernel_r)**2 + (x-kernel_r)**2) > kernel_r**2:
                kernel[y, x] = 0
    return kernel


# 生成3D球形卷积核
def create_3dkernel(kernel_r):
    """
    :param kernel_r: 指定的卷积核半径(d=2*r+1)
    :return: 距圆心kernel_r内为1，其余为0的kernel（np.uint8）
    """
    kernel_d = kernel_r * 2 + 1
    kernel = np.ones((kernel_d,kernel_d,kernel_d), np.uint8) * 255
    for z in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            for x in range(kernel.shape[2]):
                if ((z - kernel_r) ** 2 + (y - kernel_r) ** 2 + (x - kernel_r) ** 2) > kernel_r ** 2:
                    kernel[z, y, x] = 0
    return kernel



# 获取opencv-python版本
# cv2.findContours()函数返回存在差异
def get_findContours_renum():
    """
    :return: cv2.findContours()函数返回的参数个数
    """
    return_num = 0
    ver = cv2.__version__.split('.')
    opencv_ver = int(ver[0]) * 100 + int(ver[1])
    # opencv-python 2.x与不低于3.5版本均返回contours,hierarchy
    if ((opencv_ver >= 200) and (opencv_ver < 300)) or opencv_ver >= 305 :
        return_num = 2
    # opencv-python 3.0~3.4.x版本返回image,contours,hierarchy
    elif (opencv_ver >= 300) and (opencv_ver < 305):
        return_num = 3
    else:
        assert opencv_ver >= 200, "the version of opencv-python in this platform is too old!!!"

    return return_num

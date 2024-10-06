#! /usr/bin/env python
# -*- coding=utf-8 -*-
# Some basic operations related to files
# 文件相关的常用操作
# -------------------------------------------------------------
# split_filename()  获取文件名指定级数下的信息（与传入分隔字符与连接字符有关）
# split_filepath()  获取完整路径指定目录级数下的“文件名”
# get_filepaths()   获取根目录下各文件及文件完整路径(path+filename)
# get_filedirdict() 获取根目录下各子文件夹内的文件情况 {子文件夹：文件夹内的文件名}
# -------------------------------------------------------------
# Author: SEU-BME-LBMD-chl,SEU-BME-LBMD-zzy
import os
import copy
import re



# 获取文件名指定级数下的信息（与传入分隔字符与连接字符有关）
# split_filename('USM.A.B.C.DCM.avi',2,4,'.','-') = 'B-C-DCM'
def split_filename(instring, level_s, level_e, splitchar='', linkchar=''):
    """
    :param instring: 输入的文件名或文件路径（*.xxx）
    :param level_s: 截取的起始级（从后往前数,'xxx'为第1级）
    :param level_e: 截取的终止级（从后往前数）
    :param splitchar: 分隔字符
    :param linkchar: 连接字符，将各元素连接起来，此时返回字符串
    :return: 文件指定级数范围的信息（返回元素为字符串的list或字符串（linkchar）非空）
    """
    # 检查传入参数
    assert splitchar is not None, "splitchar不能为空!"
    assert level_e >= level_s, "level_e >= level_s!"
    assert level_s and level_e, "level_s与level_e不能为空！"
    assert not (isinstance(instring,list) and len(instring)>1), "请检查instring！"
    # 当传入元素为列表时，取列表第1项的字符串
    instr = copy.deepcopy(instring)
    str_temp = instr[0] if isinstance(instring,list) else instr
    str_split = re.sub(re.compile('\\\\'),'/',str_temp)
    assert level_e <= len(str_split.rsplit(
        splitchar, maxsplit=level_e)), "split级数应不高于文件中分隔字符个数！"
    # 以分隔字符划分文件名
    str_split = str_split.rsplit(sep=splitchar, maxsplit=level_s-1)[0].rsplit(splitchar, level_e)
    str_temp = copy.deepcopy(str_split[-1-(level_e-level_s):])
    # 以连接字符连接各元素
    restr = linkchar.join(str_temp) if linkchar else str_temp

    return restr


# 获取完整路径指定目录级数下的“文件名”
# split_filepath('D:/A/B/M/N',1,3) = 'B/M/N'
def split_filepath(instring, level_s, level_e, linkchar='/', splice=True):
    """
    :param instring: 输入的字符串（完整路径（包括文件名））
    :param level_s: 截取的起始级（从后往前数）
    :param level_e: 截取的终止级（从后往前数）
    :param splice: 是否使用'/'将各级路径拼接起来
    :return: 完整路径指定目录级数下的信息（字符串格式）
    """
    # 检查传入参数
    instring_new = instring[:-1] if (instring.endswith('/') or instring.endswith('\\\\')) else instring
    linkchar_new = linkchar if splice else ''
    # 路径以'/'划分
    restr = split_filename(instring_new, level_s, level_e, splitchar='/', linkchar=linkchar_new)

    return restr


# 获取根目录下各文件名及文件完整路径(path+filename)
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
    assert len(size_range)==0 or len(size_range)==2, "len(size_range) = 0 or 2!"
    assert (not suffix) or type(suffix)==str or type(suffix)==list, "suffix should be None, str or list!"
    refilenames = []   #存放各文件名
    refiledirs = []    #存放各文件路径
    refilepaths = []   #存放各文件完整路径(path+filename)
    rootdir = copy.deepcopy(indir)
    for root_dirs, file_dirs, filenames in os.walk(rootdir):
        for filename in filenames:
            # 非指定文件类型则跳过
            if suffix:
                if (type(suffix)==str) and (not filename.endswith(suffix)):
                    continue
                if type(suffix)==list:
                    str_temp = '.{}'.format(filename.rsplit(sep='.', maxsplit=1)[-1])
                    if not ((str_temp in suffix) or (str_temp[1:] in suffix)):
                        continue
            if prefix and (not filename.startswith(prefix)):
                continue
            # 非指定文件尺寸则跳过
            if len(size_range) > 0:
                file_size = os.path.getsize(os.path.join(root_dirs,filename))
                if ((file_size<size_range[0]) or (file_size>size_range[1])):
                    continue
            refilepaths.append(os.path.join(root_dirs,filename))
            refiledirs.append(os.path.join(root_dirs))
            refilenames.append(os.path.join(filename))

    if not onlypath:
        return refilepaths,refiledirs,refilenames
    else:
        return refilepaths


# 获取根目录下各子文件夹内的文件情况
def get_filedirdict(indir, prefix='', suffix='', size_range=[]):
    """
    :param indir: 文件根目录
    :param prefix: 文件前缀（如"CT."），默认不指定
    :param suffix: 文件后缀（如"*.dcm"），默认不指定
    :param size_range: 指定文件尺寸（[min,max]单位bytes）
    :return: key对应目录（文件夹）名，value为文件夹内的文件名构成的字典
    """
    # 根目录下各文件及文件完整路径
    _,filedirs,filenames = get_filepaths(indir,prefix,suffix,size_range)
    # key对应目录（文件夹）名，value为文件夹内的文件名构成的字典
    filedirdict = dict()
    for i in range(len(filedirs)):
        filedir = filedirs[i]
        if filedir not in filedirdict.keys():
            filedirdict[filedir] = []
        filedirdict[filedir].append(filenames[i])

    return filedirdict

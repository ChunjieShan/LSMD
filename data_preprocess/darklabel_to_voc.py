from fileinput import filename
from xml.dom import xmlbuilder
import cv2 as cv
import pandas as pd
import os
from xml.dom.minidom import Document


class Label2VOC:
    def __init__(self) -> None:
        pass


def convert_single_video_to_voc(video_path, 
                                anno_path,
                                img_save_path,
                                anno_save_path):
    cap = cv.VideoCapture(video_path) 
    labels_list = []
    # [ fn, id, x1, y1, w, h, c=-1, c=-1, c=-1, c=-1, cname]
    label = pd.read_csv(anno_path)
    labels_list.append(label.values[:, :].tolist())

    index = 0
    template = "img_{:05d}"
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(img_save_path, template.format(index) + ".jpg")
            labelled_frame_idx = labels_list[index, 0]

            if labelled_frame_idx == index:
                cv.imwrite(img_path) 

            

        index += 1
        

def csv_single_line_to_xml(csv_line, img_path, img_shape):
    img_path_split = img_path.split("/")
    img_root_dir = os.path.join(img_path_split[-3], img_path_split[-2])
    img_file_name = img_path_split[-1].split('.')[0]
    h, w, c = img_shape
    x, y, w, h = csv_line[2: 6]

    xml_builder = Document()
    # <annotation> </annotation>
    anno = xml_builder.createElement("annotation")
    xml_builder.appendChild(anno)

    # <folder> </folder>
    folder = xml_builder.createElement("folder") 
    folder_content = xml_builder.createTextNode(img_root_dir)
    folder.appendChild(folder_content)
    anno.appendChild(folder)

    # <filename> </filename>
    file_name = xml_builder.createElement("filename")
    file_name_content = xml_builder.createTextNode(img_file_name)
    file_name.appendChild(file_name_content)
    anno.appendChild(file_name)

    # <source> </source>
    source = xml_builder.createElement("source") 
    data_base = xml_builder.createElement("database")
    data_base_content = xml_builder.createTextNode("ILSVRC_2015")
    data_base.appendChild(data_base_content)
    source.appendChild(data_base)
    anno.appendChild(source)

    # <size> </size>
    size = xml_builder.createElement("size")
    width = xml_builder.createElement("width")
    height = xml_builder.createElement("height")

    width_content = xml_builder.createTextNode(str(w))
    height_content = xml_builder.createTextNode(str(h))

    width.appendChild(width_content)
    height.appendChild(height_content)

    size.appendChild(width)
    size.appendChild(height)

    anno.appendChild(size)
    
    # <object> </object>
    object = xml_builder.createElement("object")
    bndbox = xml_builder.createElement("bndbox")
    x_max = xml_builder.createElement("xmax")
    x_min = xml_builder.createElement("xmin")
    y_max = xml_builder.createElement("ymax")
    y_min = xml_builder.createElement("ymin")

    x_max_content = xml_builder.createTextNode(str(x + w))
    x_min_content = xml_builder.createTextNode(str(x))
    y_max_content = xml_builder.createTextNode(str(y + h))
    y_min_content = xml_builder.createTextNode(str(y))

    x_max.appendChild(x_max_content)
    x_min.appendChild(x_min_content)
    y_max.appendChild(y_max_content)
    y_min.appendChild(y_min_content)

    bndbox.appendChild()

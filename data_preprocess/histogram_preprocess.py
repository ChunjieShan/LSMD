import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalize(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    return dst


def video_eq_hist_preprocess(_video_path):
    cap = cv.VideoCapture(_video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    writer = cv.VideoWriter("./preprocessed.avi", cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                            (np.int(width), np.int(height)), False)
    while True:
        ret, frame = cap.read()
        if ret:
            eq = histogram_equalize(frame)
            writer.write(eq)
        else:
            break


if __name__ == '__main__':
    video_path = r"E:\Data\PSAX-A\original_data\USM.1.2.840.113619.2.391.3780.1637333582.109.1.512_AURORA-003780.DCM.avi"
    video_eq_hist_preprocess(video_path)
    # histogram_equalize(r"E:\Data\PSAX-A\PSAX_A_KeyFrame\ED\image_data\USM.1.2.840.113619.2.391.3780.1637144353.171.1.512_AURORA-003780.DCM\image_00014.jpg")
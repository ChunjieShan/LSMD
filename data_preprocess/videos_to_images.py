import cv2 as cv
import os

import pandas as pd


def video_to_images(videos_dir,
                    labels_dir,
                    images_dir):
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    videos_list = os.listdir(videos_dir)

    for video_name in videos_list:
        image_dir_name = video_name.split(".avi")[0]
        print("[I] Currently processing: {}, copying to {}".format(video_name, os.path.join(images_dir, image_dir_name)))
        label_path = os.path.join(labels_dir, image_dir_name + ".csv")

        df = pd.read_csv(label_path, header=None)

        frame_indexes = []
        for row in df.index:
            label_info = df.loc[row]
            frame_idx = int(label_info[0])
            if not len(frame_indexes) or (frame_idx != frame_indexes[-1]):
                frame_indexes.append(frame_idx)

        print("[I] Frame indexes: ", frame_indexes)
        video_path = os.path.join(videos_dir, video_name)
        cap = cv.VideoCapture(video_path)

        template = "img_{:05d}.jpg"
        counter = 0
        while True:
            ret, frame = cap.read()

            if ret:
                if counter in frame_indexes:
                    dest_path = os.path.join(images_dir, image_dir_name)
                    if not os.path.exists(dest_path):
                        os.mkdir(dest_path)

                    cv.imwrite(os.path.join(dest_path, template.format(counter)), frame)

                counter += 1

            else:
                break


if __name__ == '__main__':
    video_path = "/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/20230201/20230201_cross/videos"
    labels_path = "/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/20230201/20230201_cross/labels"
    images_path = "/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/20230201/20230201_cross/images"
    video_to_images(videos_dir=video_path,
                    labels_dir=labels_path,
                    images_dir=images_path)

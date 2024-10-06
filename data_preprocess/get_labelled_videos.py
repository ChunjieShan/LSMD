import os
import shutil


def get_labelled_videos(video_dir, label_dir, des_dir, video_suffix=".avi", label_suffix=".csv"):
    file_names = []
    for file in os.listdir(label_dir):
        file_name = file.split(label_suffix)[0]
        file_names.append(file_name)

    unlabelled_videos = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            video_file_name = file.split(video_suffix)[0]
            for file_name in file_names:
                if video_file_name == file_name:
                    unlabelled_videos.append(os.path.join(root, file))
                    print(os.path.join(root, file))
                    shutil.copy(os.path.join(root, file), des_dir)


def get_unlabelled_video(video_dir, label_dir, des_dir, video_suffix=".avi", label_suffix=".avi"):
    labelled_file_names = []
    for file in os.listdir(label_dir):
        file_name = file.split(label_suffix)[0]
        labelled_file_names.append(file_name)

    unlabelled_videos = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            video_file_name = file.split(video_suffix)[0]
            if video_file_name not in labelled_file_names:
                unlabelled_videos.append(os.path.join(root, file))
                print(os.path.join(root, file))

                shutil.copy(os.path.join(root, file), des_dir)


if __name__ == '__main__':
    video_path = "/home/scj/Projects/Code/Data/PSAX-A/PSAX_OD/New/original_data"
    # label_path = "E:\\Data\\PSAX-A\\PSAX_OD\\PSAXA_selected_label\\object_detect\\PSAXA_LV\\video_label"
    label_path = "/home/scj/Projects/Code/Data/PSAX-A/PSAX_OD/New/labels"
    des_path = "/home/scj/Projects/Code/Data/PSAX-A/PSAX_OD/New/video_data"
    get_labelled_videos(video_path, label_path, des_path)

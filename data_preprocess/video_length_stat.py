import cv2 as cv
import os

import numpy as np


def get_video_duration(video_path):
    """获取视频时长（单位：秒）"""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    # 获取视频的帧数
    frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    # 获取视频的帧率
    fps = cap.get(cv.CAP_PROP_FPS)
    # 计算视频时长
    duration = frames / fps
    cap.release()
    return duration


def group_videos_by_duration(directory):
    """遍历目录中的所有视频文件并按时长分组"""
    # 初始化分组字典
    groups = {'<5s': [], '5s~15s': [], '>15s': []}

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 添加需要支持的视频格式
            video_path = os.path.join(directory, filename)
            duration = get_video_duration(video_path)

            if duration is not None:
                if duration < 5:
                    groups['<5s'].append([filename, duration])
                elif 5 <= duration <= 15:
                    groups['5s~15s'].append([filename, duration])
                else:
                    groups['>15s'].append([filename, duration])

    return groups


if __name__ == '__main__':
    video_root = "/mnt/j/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20240725/videos/external_test/"
    # video_root = "/mnt/j/Dataset/BUV/videos"
    groups = group_videos_by_duration(video_root)
    total_lengths = []
    for group, count in groups.items():
        video_lengths = [video_info[1] for video_info in count]
        total_lengths.extend(video_lengths)
        max_length, min_length = max(video_lengths), min(video_lengths)
        mean_length, std_length = np.mean(np.array(video_lengths)), np.std(np.array(video_lengths))
        print(f"{group}: {len(count)} {min_length}~{max_length} mean±std: {mean_length}±{std_length}")

    total_mean, total_std = np.mean(np.array(total_lengths)), np.std(np.array(total_lengths))
    print(f"mean±std: {total_mean}±{total_std}")

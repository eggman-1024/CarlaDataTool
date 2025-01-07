import numpy as np
import os
from tqdm import tqdm

label_path = r"F:\Software\carla_tools\dataset\record\training\label_2"
# label_path = r"Q:\kitti\training\label_2"

def count_difficulty(label_path):
    """
    根据type统计数据集中各难度的数量
    :param label_path: str, 标签文件夹路径
    :return: None
    """
    filenames = sorted(os.listdir(label_path))
    type_dict = {"DontCare": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Car": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Van": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Truck": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Pedestrian": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Person_sitting": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Cyclist": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Tram": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Misc": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0}}
    for filename in tqdm(filenames, desc="Counting difficulty", colour="green"):
        with open(os.path.join(label_path, filename), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                if line[0] in type_dict:
                    type_dict[line[0]][line[2]] += 1
    print("="*100)
    print(label_path)
    print(type_dict)


def count_difficulty_by_ImageSets(label_path, ImageSets_path=r"F:\Software\carla_tools\dataset\record\sample_ImageSets"):
    """
    根据ImageSets文件夹统计数据集中各难度的数量
    :param label_path: str, 标签文件夹路径
    :param ImageSets_path: str, ImageSets文件夹路径
    :return: None
    """
    filenames = sorted(os.listdir(label_path))
    type_dict = {"DontCare": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Car": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Van": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Truck": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Pedestrian": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Person_sitting": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Cyclist": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Tram": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0},
                 "Misc": {"-1": 0, "0": 0, "1": 0, "2": 0, "3": 0}}

    indices = []
    with open(os.path.join(ImageSets_path, "trainval.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            indices.append(int(line.strip()))
    for index in tqdm(indices, desc="Counting difficulty by ImageSets", colour="green"):
        with open(os.path.join(label_path, f"{index:0>6d}.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                if line[0] in type_dict:
                    type_dict[line[0]][line[2]] += 1
    print("="*100)
    print(label_path)
    print(type_dict)

if __name__ == "__main__":
    # count_difficulty(label_path)
    count_difficulty_by_ImageSets(label_path)
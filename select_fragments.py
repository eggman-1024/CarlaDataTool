import os
import argparse
import numpy as np
import shutil
from tqdm import tqdm

def select_fragments(root_dir, save_dir, num, train=True):
    """
    选择数据集中的片段数据
    :param root_dir: str, 数据集的路径
    :param save_dir: str, 保存路径
    :param num: int, 保存的数据数量
    :return: None
    """
    if train:
        root_dir = os.path.join(root_dir, "training")
        save_dir = os.path.join(save_dir, "training")
        os.makedirs(save_dir, exist_ok=True)
    else:
        root_dir = os.path.join(root_dir, "testing")
        save_dir = os.path.join(save_dir, "testing")

    calib_dir = os.path.join(root_dir, "calib")
    image_2_dir = os.path.join(root_dir, "image_2")
    if train:
        label_2_dir = os.path.join(root_dir, "label_2")
    status_dir = os.path.join(root_dir, "status")
    velodyne_dir = os.path.join(root_dir, "velodyne")

    if train:
        assert len(os.listdir(calib_dir)) == len(os.listdir(image_2_dir)) == len(os.listdir(label_2_dir)) == len(os.listdir(status_dir)) == len(os.listdir(velodyne_dir))
    else:
        assert len(os.listdir(calib_dir)) == len(os.listdir(image_2_dir)) == len(os.listdir(status_dir)) == len(os.listdir(velodyne_dir))



    # 初始化一个空array,存放需要copy的[start_idx, save_idx]
    idx_array = np.zeros((0, 2), dtype=int)
    print(f"idx_array.shape:{idx_array.shape[0]}")

    # 获取root_dir下的所有文件
    root_files = sorted(os.listdir(calib_dir))
    idx_num = len(root_files)

    # 循环获取用户命令行输入
    while True:
        print("请输入需要选择的数据片段的起始和截取数量,输入q退出")
        print(f"当前已存储的数据片段:{list(idx_array)}")
        print("例如:0 40")
        input_str = input()
        if input_str == "q":
            break
        input_list = input_str.split()
        if len(input_list) not in [1, 2]:
            print("输入的数量不合法")
            continue

        if len(input_list) == 1:
            save_num = num
        else:
            save_num = int(input_list[1])

        # 判断输入是否合法
        start_idx = int(input_list[0])
        if start_idx < 0 or start_idx >= idx_num:
            print("输入的起始索引不合法")
            continue

        if idx_array.shape[0] != 0:
            last_save_idx = idx_array[-1, 0] + idx_array[-1, 1]
            if start_idx < last_save_idx:
                print("输入的起始索引不合法")
                continue
        
        # 如果剩下的数量不够了
        if start_idx + save_num > idx_num:
            save_num = idx_num - start_idx + 1
    
        idx_array = np.vstack((idx_array, np.array([start_idx, save_num])))

    # 初始化保存的索引
    save_index = 0
    root_index_list = []
    for idx in idx_array:
        start_idx, save_num = idx
        root_index_list += list(range(start_idx, start_idx + save_num))

    for root_index in tqdm(root_index_list, desc="Copying data", colour="green"):
        calib_file = os.path.join(calib_dir, "{:06d}.txt".format(root_index))
        image_2_file = os.path.join(image_2_dir, "{:06d}.png".format(root_index))
        status_file = os.path.join(status_dir, "{:06d}.txt".format(root_index))
        velodyne_file = os.path.join(velodyne_dir, "{:06d}.bin".format(root_index))

        save_calib_file = os.path.join(save_dir, "calib", "{:06d}.txt".format(save_index))
        save_image_2_file = os.path.join(save_dir, "image_2", "{:06d}.png".format(save_index))
        save_status_file = os.path.join(save_dir, "status", "{:06d}.txt".format(save_index))
        save_velodyne_file = os.path.join(save_dir, "velodyne", "{:06d}.bin".format(save_index))

        os.makedirs(os.path.join(save_dir, "calib"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "image_2"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "status"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "velodyne"), exist_ok=True)

        shutil.copyfile(calib_file, save_calib_file)
        shutil.copyfile(image_2_file, save_image_2_file)
        shutil.copyfile(status_file, save_status_file)
        shutil.copyfile(velodyne_file, save_velodyne_file)

        if train:
            label_2_file = os.path.join(label_2_dir, "{:06d}.txt".format(root_index))
            save_label_2_file = os.path.join(save_dir, "label_2", "{:06d}.txt".format(save_index))
            os.makedirs(os.path.join(save_dir, "label_2"), exist_ok=True)
            shutil.copyfile(label_2_file, save_label_2_file)

        save_index += 1

def split_train_val(save_dir, train_val_ratio=0.55):
    ImageSets_file = os.path.join(save_dir, "ImageSets")
    os.makedirs(ImageSets_file, exist_ok=True)

    # 获取所有的数据数量
    calib_dir = os.path.join(save_dir, "training", "calib")
    for file in os.listdir(calib_dir):
        idx = file.split(".")[0]
        if int(idx) < train_val_ratio * len(os.listdir(calib_dir)):
            with open(os.path.join(ImageSets_file, "train.txt"), "a") as f:
                f.write(idx + "\n")
        else:
            with open(os.path.join(ImageSets_file, "val.txt"), "a") as f:
                f.write(idx + "\n")
        with open(os.path.join(ImageSets_file, "trainval.txt"), "a") as f:
            f.write(idx + "\n")


def main(args):
    select_fragments(args.root_dir, args.save_dir, args.num, args.train)
    split_train_val(args.save_dir)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Select fragments of dataset")
    argparser.add_argument("--root_dir", "-r", default= r"F:\Software\carla_tools\dataset\record_2025_0106_2019\vehicle.tesla.model3.master\kitti_object", type=str, help="The root directory of dataset")
    argparser.add_argument("--save_dir", "-s", default= r"F:\Software\carla_tools\dataset\record", type=str, help="The save directory of selected fragments")
    argparser.add_argument("--num", "-n", default= 40, type=int, help="The number of selected fragments")
    argparser.add_argument("--train", default=True, type=bool, help="Training or Testing dataset")

    args = argparser.parse_args()
    main(args) 
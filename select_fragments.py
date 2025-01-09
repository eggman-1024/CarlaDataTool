import os
import argparse
import numpy as np
import shutil
from tqdm import tqdm

def select_fragments(root_dir, save_dir, clip_length, train=True):
    """
    选择数据集中的片段数据
    :param root_dir: str, 数据集的路径
    :param save_dir: str, 保存路径
    :param num: int, 截取的片段帧数
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



    # 初始化一个空array,存放需要copy的[start_idx, save_idx, end_idx+1]
    idx_array = np.zeros((0, 3), dtype=int)

    # 获取root_dir下的所有文件
    root_files = sorted(os.listdir(calib_dir))
    idx_num = len(root_files)

    # 循环获取用户命令行输入
    while True:
        print("=================================================================================")
        print(f"已添加片段数:{len(list(idx_array))}")
        print(f"当前已存储的数据片段:{list(idx_array)}")
        print(f"当前已存储的数据片段数量:{list(idx_array)[-1][0] + list(idx_array)[-1][1] if len(list(idx_array)) > 0 else 0}")
        print("请输入需要选择的数据片段的起始id:\n[输入q退出/输入s保存/d删除最后一个片段]")

        input_str = input()
        if input_str == "q":
            return
        if input_str == "s":
            break
        if input_str == "d":
            if idx_array.shape[0] == 0:
                continue
            idx_array = idx_array[:-1]
            continue

        input_list = input_str.split()
        if len(input_list) != 1:
            print("输入的数量不合法")
            continue


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
        if start_idx + clip_length > idx_num:
            clip_length = idx_num - start_idx + 1
    
        idx_array = np.vstack((idx_array, np.array([start_idx, clip_length, start_idx + clip_length])))

    # 初始化保存的索引
    os.makedirs(os.path.join(save_dir, "calib"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "image_2"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "status"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "velodyne"), exist_ok=True)

    cache_index = len(os.listdir(os.path.join(save_dir, "calib")))
    save_index = cache_index
    root_index_list = []
    for idx in idx_array:
        start_idx, save_num, end_idx = idx
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

def split_train_val(save_dir, clip_length, train_val_ratio=0.55):
    ImageSets_file = os.path.join(save_dir, "ImageSets")
    os.makedirs(ImageSets_file, exist_ok=True)

    # 获取所有的数据数量
    calib_dir = os.path.join(save_dir, "training", "calib")
    # 80帧为一个片段，在划分训练集和验证集时，以片段为单位
    clip_num = len(os.listdir(calib_dir)) // clip_length
    train_clip_idx_list = []
    val_clip_idx_list = []

    for i in range(clip_num):
        if i <= clip_num * train_val_ratio:
            train_clip_idx_list.append(i)
        else:
            val_clip_idx_list.append(i)

    # 根据片段的索引，生成训练集和验证集的索引
    train_index_list = []
    val_index_list = []
    for clip_idx in train_clip_idx_list:
        train_index_list += list(range(clip_idx * clip_length, clip_idx * clip_length + clip_length))
    for clip_idx in val_clip_idx_list:
        val_index_list += list(range(clip_idx * clip_length, clip_idx * clip_length + clip_length))

    # train.txt
    with open(os.path.join(ImageSets_file, "train.txt"), "w") as f:
        for idx in train_index_list:
            f.write("{:06d}\n".format(idx))
    # val.txt
    with open(os.path.join(ImageSets_file, "val.txt"), "w") as f:
        for idx in val_index_list:
            f.write("{:06d}\n".format(idx))
    # trainval.txt
    with open(os.path.join(ImageSets_file, "trainval.txt"), "w") as f:
        for idx in train_index_list + val_index_list:
            f.write("{:06d}\n".format(idx))


def main(args):
    select_fragments(args.root_dir, args.save_dir, args.clip_length, args.train)
    # split_train_val(args.save_dir, args.clip_length)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Select fragments of dataset")
    argparser.add_argument("--root_dir", "-r", default= r"F:\Software\carla_tools\dataset\record_2025_0108_2211\vehicle.tesla.model3.master\kitti_object", type=str, help="The root directory of dataset")
    argparser.add_argument("--save_dir", "-s", default= r"F:\Software\carla_tools\dataset\record", type=str, help="The save directory of selected fragments")
    argparser.add_argument("--clip_length", "-l", default=80, type=int, help="The length of selected fragments")
    argparser.add_argument("--train", default=True, type=bool, help="Training or Testing dataset")

    args = argparser.parse_args()
    main(args) 
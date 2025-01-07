import os
import shutil
from tqdm import tqdm

from param import RAW_DATA_PATH, DATASET_PATH

INDEX_TRAIN = 0
INDEX_TEST = 0

def copy_files(dataset_dir, save_dir, index, map_index, type="training"):
    """
    复制文件
    """
    global INDEX_TRAIN, INDEX_TEST
    if type == "training":
        train_dir = os.path.join(dataset_dir, "training")

        calib_files = sorted(os.listdir(os.path.join(train_dir, "calib")))
        image_files = sorted(os.listdir(os.path.join(train_dir, "image_2")))
        label_files = sorted(os.listdir(os.path.join(train_dir, "label_2")))
        status_files = sorted(os.listdir(os.path.join(train_dir, "status")))
        velodyne_files = sorted(os.listdir(os.path.join(train_dir, "velodyne")))

        assert len(calib_files) == len(image_files) == len(label_files) == len(status_files) == len(velodyne_files)
        for j in tqdm(range(len(calib_files)), desc=f"Copying data from {dataset_dir}", colour="green"):
            new_index = "{:0>6d}".format(INDEX_TRAIN)
            shutil.copy(os.path.join(train_dir, "calib", calib_files[j]), os.path.join(save_dir, "calib", f"{new_index}.txt"))
            shutil.copy(os.path.join(train_dir, "image_2", image_files[j]), os.path.join(save_dir, "image_2", f"{new_index}.png"))
            shutil.copy(os.path.join(train_dir, "label_2", label_files[j]), os.path.join(save_dir, "label_2", f"{new_index}.txt"))
            shutil.copy(os.path.join(train_dir, "status", status_files[j]), os.path.join(save_dir, "status", f"{new_index}.txt"))
            shutil.copy(os.path.join(train_dir, "velodyne", velodyne_files[j]), os.path.join(save_dir, "velodyne", f"{new_index}.bin"))
            with open(os.path.join(save_dir, "map_index.txt"), 'a') as f:
                f.write(f"{map_index} {new_index}\n")
            INDEX_TRAIN += 1

    elif type == "testing":
        test_dir = os.path.join(dataset_dir, "testing")

        calib_files = sorted(os.listdir(os.path.join(test_dir, "calib")))
        image_files = sorted(os.listdir(os.path.join(test_dir, "image_2")))
        status_files = sorted(os.listdir(os.path.join(test_dir, "status")))
        velodyne_files = sorted(os.listdir(os.path.join(test_dir, "velodyne")))

        assert len(calib_files) == len(image_files) == len(status_files) == len(velodyne_files)
        for j in tqdm(range(len(calib_files)), desc=f"Copying data from {dataset_dir}", colour="green"):
            new_index = "{:0>6d}".format(INDEX_TEST)
            shutil.copy(os.path.join(test_dir, "calib", calib_files[j]), os.path.join(save_dir, "calib", f"{new_index}.txt"))
            shutil.copy(os.path.join(test_dir, "image_2", image_files[j]), os.path.join(save_dir, "image_2", f"{new_index}.png"))
            shutil.copy(os.path.join(test_dir, "status", status_files[j]), os.path.join(save_dir, "status", f"{new_index}.txt"))
            shutil.copy(os.path.join(test_dir, "velodyne", velodyne_files[j]), os.path.join(save_dir, "velodyne", f"{new_index}.bin"))
            with open(os.path.join(save_dir, "map_index.txt"), 'a') as f:
                f.write(f"{map_index} {new_index}\n")
            INDEX_TEST += 1

def merge_multi_maps(dataset_dir_list, save_dir, train_ratio=0.55):
    """
    合并在多个地图采集的数据集
    """
    for i, dataset_dir in tqdm(enumerate(dataset_dir_list), desc="Merging datasets", colour="yellow"):
        map_index = f"{i:0>2d}"
        
        # 创建目标文件夹
        new_train_dir = os.path.join(save_dir, "training")
        new_test_dir = os.path.join(save_dir, "testing")
        sub_dirs = ['calib', 'image_2', 'label_2', 'status', 'velodyne']
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(new_train_dir, sub_dir), exist_ok=True)
            if sub_dir != 'label_2':
                os.makedirs(os.path.join(new_test_dir, sub_dir), exist_ok=True)
        # 合并training和testing数据集
        copy_files(dataset_dir, new_train_dir, INDEX_TRAIN, map_index, type="training")
        # copy_files(dataset_dir, new_test_dir, INDEX_TEST, map_index, type="testing")

        # # 生成ImageSets文件
        # for i in range(INDEX_TRAIN):
        #     if i < INDEX_TRAIN * 0.8:
        #         with open(os.path.join(save_dir, "ImageSets", "train.txt"), 'a') as f:
        #             f.write(f"{i:0>6d}\n")
        #     else:
        #         with open(os.path.join(save_dir, "ImageSets", "val.txt"), 'a') as f:
        #             f.write(f"{i:0>6d}\n")
        #     with open(os.path.join(save_dir, "ImageSets", "trainval.txt"), 'a') as f:
        #         f.write(f"{i:0>6d}\n")

        # 选取每个地图的一半数据作为training数据集
        train_set = []
        val_set = []    
        with open(os.path.join(new_train_dir, "map_index.txt"), 'r') as f:
            map_index_list = f.readlines()
            map_index_list = [line.strip() for line in map_index_list] # 去除换行符
            # 统计不同map_index的数据量
            map_index_dict = {}
            for line in map_index_list:
                map_id, index = line.split(' ')
                if map_id in map_index_dict:
                    map_index_dict[map_id] += 1
                else:
                    map_index_dict[map_id] = 1

            # 对每个场景进行train-val划分
            for j, line in enumerate(map_index_list):
                map_index, index = line.split(' ')
                if int(index) % map_index_dict[map_index] < map_index_dict[map_index] * train_ratio:
                    # 若地图场景是偶数，将其加入train_set
                    if int(map_index) % 2 == 0:
                        train_set.append(index)
                    # 若地图场景是奇数，将其加入val_set
                    else:
                        val_set.append(index)
                else:
                    if int(map_index) % 2 == 0:
                        val_set.append(index)
                    else:
                        train_set.append(index)
        if not os.path.exists(os.path.join(save_dir, "ImageSets")):
            os.makedirs(os.path.join(save_dir, "ImageSets"))
        # 写入train.txt和val.txt
        with open(os.path.join(save_dir, "ImageSets", "train.txt"), 'w') as f:
            for index in train_set:
                f.write(f"{index}\n")
        with open(os.path.join(save_dir, "ImageSets", "val.txt"), 'w') as f:
            for index in val_set:
                f.write(f"{index}\n")
        # 写入trainval.txt
        with open(os.path.join(save_dir, "ImageSets", "trainval.txt"), 'w') as f:
            trainval_set = train_set + val_set
            for index in trainval_set:
                f.write(f"{index}\n")


if __name__ == "__main__":
    dataset_dir_list = [r"dataset\record_2025_0106_2019\vehicle.tesla.model3.master\kitti_object", 
                        r"dataset\record_2025_0106_2025\vehicle.tesla.model3.master\kitti_object", 
                        r"dataset\record_2025_0106_2032\vehicle.tesla.model3.master\kitti_object", 
                        r"dataset\record_2025_0106_2038\vehicle.tesla.model3.master\kitti_object",
                        r"dataset\record_2025_0106_2045\vehicle.tesla.model3.master\kitti_object",
                        r"dataset\record_2025_0106_2052\vehicle.tesla.model3.master\kitti_object",]
    save_dir = r"{}\carla4d_zrt".format(DATASET_PATH)
    merge_multi_maps(dataset_dir_list, save_dir)
            

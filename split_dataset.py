import os
import shutil
import numpy as np
import glob
from tqdm import tqdm
import pandas as pd

def get_rotation_matrix(yaw, roll, pitch):
    # Convert degrees to radians if necessary
    yaw, roll, pitch = np.radians([yaw, roll, pitch])
    
    # Roll matrix (rotation around X-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch matrix (rotation around Y-axis)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw matrix (rotation around Z-axis)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations (Yaw → Pitch → Roll)
    R = R_z @ R_y @ R_x
    return R

def generate_image_sets(path_to_kitti_object: str, type: str, val_ratio: float = 0.25):
    calib_data_list = sorted(glob.glob(f"{path_to_kitti_object}/{type}/calib/*.txt"))
    image_sets = []
    for item in calib_data_list:
        frame_id = os.path.splitext(os.path.split(item)[-1])[0]
        image_sets.append(f"{frame_id}\n")
    image_sets_dir = f"{path_to_kitti_object}/ImageSets"
    os.makedirs(image_sets_dir, exist_ok=True)
    if type == 'training':
        val_size = int(len(image_sets) * val_ratio)
        # random shuffle
        # np.random.shuffle(image_sets)
        val_set = image_sets[:val_size]
        train_set = image_sets[val_size:]

        with open(f"{image_sets_dir}/train.txt", 'w') as file:
            file.writelines(train_set)
        with open(f"{image_sets_dir}/val.txt", 'w') as file:
            file.writelines(val_set)
        with open(f"{image_sets_dir}/trainval.txt", 'w') as file:
            file.writelines(image_sets)

    elif type == 'testing':
        with open(f"{image_sets_dir}/test.txt", 'w') as file:
            file.writelines(image_sets)


def split_dataset(raw_data_root, dataset_root, new_dataset_root, train_ratio=0.8):
    train_dir = os.path.join(dataset_root, 'training')
    test_dir = os.path.join(dataset_root, 'testing')

    new_train_dir = os.path.join(new_dataset_root, 'training')
    new_test_dir = os.path.join(new_dataset_root, 'testing')

    # 创建目标文件夹
    sub_dirs = ['calib', 'image_2', 'label_2', 'velodyne']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(new_train_dir, sub_dir), exist_ok=True)
        os.makedirs(os.path.join(new_test_dir, sub_dir), exist_ok=True)

    # 获取训练数据文件列表
    calib_files = sorted(os.listdir(os.path.join(train_dir, 'calib')))
    image_files = sorted(os.listdir(os.path.join(train_dir, 'image_2')))
    label_files = sorted(os.listdir(os.path.join(train_dir, 'label_2')))
    velodyne_files = sorted(os.listdir(os.path.join(train_dir, 'velodyne')))

    # 将velodyne_files中的点云的v_x,v_y从世界坐标系转为自车雷达坐标系
    # 读取csv文件除开头外每一行
    vehicle_status = pd.read_csv(os.path.join(raw_data_root, 'vehicle_status.csv'))
    for i in tqdm(range(len(velodyne_files)), desc="Converting v_x, v_y of pointcloud to ego coordinate system"):
        # 读取csv文件中的第i行
        row = vehicle_status.iloc[i]
        # 获取第i行的roll pitch yaw
        roll, pitch, yaw = row['roll'], row['pitch'], row['yaw']
        # 构建3×3旋转矩阵
        R = get_rotation_matrix(yaw, roll, pitch)
        # 将点云数据中的vx,vy转为自车雷达坐标系
        # 旋转矩阵的逆矩阵
        R_inv = np.linalg.inv(R)
        # 旋转矩阵的逆矩阵的转置
        R_inv_T = R_inv.T
        # 将点云数据中的vx,vy转为自车雷达坐标系
        # 读取点云数据
        data = np.fromfile(os.path.join(train_dir, 'velodyne', velodyne_files[i]), dtype=np.float32).reshape(-1, 8)
        # 读取点云数据中的vx,vy
        vx = data[:, 5]
        vy = data[:, 6]
        # 计算自车雷达坐标系下的vx,vy
        lidar_coor_vx = R_inv_T[0, 0] * vx + R_inv_T[0, 1] * vy
        lidar_coor_vy = R_inv_T[1, 0] * vx + R_inv_T[1, 1] * vy
        data[:, 5] -= lidar_coor_vx
        data[:, 6] -= lidar_coor_vy
        data.tofile(os.path.join(train_dir, 'velodyne', velodyne_files[i]))
    

    # # 检查label_files中每一个文件中的每一行，如果为空，删除该文件索引
    # deprecated_files = []
    # for i in range(len(label_files)):
    #     with open(os.path.join(train_dir, 'label_2', label_files[i]), 'r') as f:
    #         lines = f.readlines()
    #         flag = False
    #         # 读取每一行第一个单词，并判断是否是"Car","Pedestrian","Cyclist"
    #         for line in lines:
    #             if line.split(' ')[0] in ["Car", "Pedestrian", "Cyclist"]:
    #                 flag = True
    #                 break
    #         if not flag:
    #             deprecated_files.append(i)
    # # 删除label_files中的空文件
    # for i in deprecated_files:
    #     os.remove(os.path.join(train_dir, 'calib', calib_files[i]))
    #     os.remove(os.path.join(train_dir, 'image_2', image_files[i]))
    #     os.remove(os.path.join(train_dir, 'label_2', label_files[i]))
    #     os.remove(os.path.join(train_dir, 'velodyne', velodyne_files[i]))

    # # 再次获取训练数据文件列表
    # calib_files = sorted(os.listdir(os.path.join(train_dir, 'calib')))
    # image_files = sorted(os.listdir(os.path.join(train_dir, 'image_2')))
    # label_files = sorted(os.listdir(os.path.join(train_dir, 'label_2')))
    # velodyne_files = sorted(os.listdir(os.path.join(train_dir, 'velodyne')))

    # 确保四个文件夹中的文件数量相同
    assert len(calib_files) == len(image_files) == len(label_files) == len(velodyne_files)

    # # seed
    # np.random.seed(42)
    # # 打乱文件列表
    # indices = np.random.permutation(len(calib_files))

    # 计算切分索引
    indices = np.arange(len(calib_files))
    total_files = len(indices)
    test_size = int(total_files * (1 - train_ratio))  # 计算测试集大小
    train_size = total_files - test_size  # 计算训练集大小

    # 分配文件
    for i in tqdm(range(total_files)):
        if i < train_size:
            # 重新编号并复制到 training
            new_index = "{:0>6d}".format(i)
            shutil.copy(os.path.join(train_dir, 'calib', calib_files[indices[i]]), os.path.join(new_train_dir, 'calib', f"{new_index}.txt"))
            shutil.copy(os.path.join(train_dir, 'image_2', image_files[indices[i]]), os.path.join(new_train_dir, 'image_2', f"{new_index}.png"))
            shutil.copy(os.path.join(train_dir, 'label_2', label_files[indices[i]]), os.path.join(new_train_dir, 'label_2', f"{new_index}.txt"))
            shutil.copy(os.path.join(train_dir, 'velodyne', velodyne_files[indices[i]]), os.path.join(new_train_dir, 'velodyne', f"{new_index}.bin"))
        else:
            # 重新编号并复制到 test
            new_index = "{:0>6d}".format(i - train_size)
            shutil.copy(os.path.join(train_dir, 'calib', calib_files[indices[i]]), os.path.join(new_test_dir, 'calib', f"{new_index}.txt"))
            shutil.copy(os.path.join(train_dir, 'image_2', image_files[indices[i]]), os.path.join(new_test_dir, 'image_2', f"{new_index}.png"))
            shutil.copy(os.path.join(train_dir, 'label_2', label_files[indices[i]]), os.path.join(new_test_dir, 'label_2', f"{new_index}.txt"))
            shutil.copy(os.path.join(train_dir, 'velodyne', velodyne_files[indices[i]]), os.path.join(new_test_dir, 'velodyne', f"{new_index}.bin"))

    print("数据分配完成！")
    types = ['training', 'testing']
    for type in types:
        generate_image_sets(new_dataset_root, type)

if __name__ == "__main__":
    raw_data_path = r"raw_data\record_2024_1025_0902\vehicle.tesla.model3.master"
    new_data_path = r"dataset\record_2024_1025_0902\vehicle.tesla.model3.master"
    split_dataset(raw_data_path, os.path.join(new_data_path, "kitti_object"), os.path.join(new_data_path, "carla4d_kitti"))
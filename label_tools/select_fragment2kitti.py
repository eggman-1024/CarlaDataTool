#!/usr/bin/python3
import os
import argparse
from codecs import ignore_errors
import math
import pickle
import sys
import time
from multiprocessing.pool import Pool as ThreadPool
from pathlib import Path
import shutil 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import RAW_DATA_PATH, DATASET_PATH
from label_tools.kitti_object.kitti_object_data_loader import *
from label_tools.kitti_object.kitti_object_helper import *




def select_fragments(root_dir, clip_length):
    """
    选择数据集中的片段数据
    :param root_dir: str, 数据集的路径
    :param save_dir: str, 保存路径
    :param num: int, 截取的片段帧数
    :return: None
    """

    # 初始化一个空array,存放需要copy的[start_idx, save_idx, end_idx+1]
    idx_array = np.zeros((0, 3), dtype=int)

    # 获取root_dir下的所有文件
    root_files = sorted(os.listdir(root_dir))
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


    # 生成需要保存的索引
    root_index_list = []
    for idx in idx_array:
        start_idx, save_num, end_idx = idx
        root_index_list += list(range(start_idx, start_idx + save_num))
    return root_index_list


def split_train_val(save_dir, clip_length, train_val_ratio=0.55):
    ImageSets_file = os.path.join(save_dir, "ImageSets")
    os.makedirs(ImageSets_file, exist_ok=True)

    # 获取所有的数据数量
    calib_dir = os.path.join(save_dir, "training", "calib")
    # [clip_length]帧为一个片段，在划分训练集和验证集时，以片段为单位
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
        for idx in tqdm(train_index_list, desc="Write train.txt"):
            f.write("{:06d}\n".format(idx))
    # val.txt
    with open(os.path.join(ImageSets_file, "val.txt"), "w") as f:
        for idx in tqdm(val_index_list, desc="Write val.txt"):
            f.write("{:06d}\n".format(idx))
    # trainval.txt
    with open(os.path.join(ImageSets_file, "trainval.txt"), "w") as f:
        for idx in tqdm(train_index_list + val_index_list, desc="Write trainval.txt"):
            f.write("{:06d}\n".format(idx))



def gather_rawdata_to_dataframe(record_name: str, vehicle_name: str, lidar_path: str, camera_path: str, cache_index: int, selected_frame_list: list):
    '''
    Gather rawdata to a pandas dataframe
    :param record_name: str, record name
    :param vehicle_name: str, vehicle name
    :param lidar_path: str, lidar path
    :param camera_path: str, camera path
    :param cache_index: int, cache index
    :param selected_frame_list: list, selected frame list
    :return: pd.DataFrame with columns ['index', 'frame', 'timestamp', 'vehicle_pose', 'vehicle_status', 
                                        'object_labels_path', 'lidar_rawdata_path', 'lidar_pose', 
                                        'camera_rawdata_path', 'camera_pose', 'camera_matrix', 'cache_index']
    '''
    rawdata_frames_df = pd.DataFrame()
    vehicle_poses_df = load_vehicle_pose("{}/{}/{}".format(RAW_DATA_PATH, record_name, vehicle_name), selected_frame_list)
    rawdata_frames_df = vehicle_poses_df
    rawdata_frames_df = rawdata_frames_df.reset_index(drop=False)

    object_labels_path_df = load_object_labels("{}/{}/others.world_0".format(RAW_DATA_PATH, record_name), selected_frame_list)
    rawdata_frames_df = pd.merge(rawdata_frames_df, object_labels_path_df, how='outer', on='frame')

    lidar_rawdata_df = load_lidar_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}/{lidar_path}", selected_frame_list)
    rawdata_frames_df = pd.merge(rawdata_frames_df, lidar_rawdata_df, how='outer', on='frame')

    camera_rawdata_path_df = load_camera_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}/{camera_path}", selected_frame_list)
    rawdata_frames_df = pd.merge(rawdata_frames_df, camera_rawdata_path_df, how='outer', on='frame')

    # cache_index用于标记保存文件的索引
    rawdata_frames_df['cache_index'] = cache_index + np.arange(len(rawdata_frames_df))

    return rawdata_frames_df




class KittiObjectLabelTool:
    def __init__(self, rawdata_df: pd.DataFrame, output_dir=None, clip_length=80):
        self.rawdata_df = rawdata_df[rawdata_df.camera_pose.notna()]
        self.range_max = 120.0
        self.range_min = 1.0
        self.points_min = 10
        self.output_dir = output_dir
        self.clip_length = clip_length

    def process(self):

        debug = False #TODO:!!
        if not debug:
            start = time.time()
            thread_pool = ThreadPool()
            thread_pool.starmap(self.process_frame, self.rawdata_df.iterrows())
            thread_pool.close()
            thread_pool.join()
            print(f"====================ImageSets====================")
            split_train_val(self.output_dir, self.clip_length)
            print("Cost: {:0<3f}s".format(time.time() - start))
        else:
            start = time.time()
            for index, frame in self.rawdata_df.iterrows():
                # index = 1120
                # frame = self.rawdata_df.iloc[index]
                self.process_frame(index, frame)
            print(f"====================ImageSets====================")
            split_train_val(self.output_dir, self.clip_length)
            print("Cost: {:0<3f}s".format(time.time()-start))



    def process_frame(self, index, frame):
        index = "{:0>6d}".format(index)
        frame_id = "{:0>6d}".format(frame['frame'])
        cache_id = "{:0>6d}".format(frame['cache_index'])
        timestamp = frame['timestamp']
        vehicle_status = frame['vehicle_status']
        lidar_trans: Transform = frame['lidar_pose']
        cam_trans: Transform = frame['camera_pose']
        cam_mat = np.asarray(frame['camera_matrix']).reshape(3, 3)
        image = read_image(frame['camera_rawdata_path'])

        # Transform matrix to transform object from lidar to camera coordinate
        T_lc = np.matmul(cam_trans.get_inverse_matrix(), lidar_trans.get_matrix())

        pointcloud_raw = read_pointcloud(frame['lidar_rawdata_path'])
        # convert v to lidar coordinate system
        pointcloud_raw = convert_v_to_lidar_coordinate(pointcloud_raw, lidar_trans)

        pointcloud_with_id_label = np.array([pointcloud_raw['x'], pointcloud_raw['y'], pointcloud_raw['z'], pointcloud_raw['id'], pointcloud_raw['label']],dtype='float64').T

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(np.array([pointcloud_raw['x'], pointcloud_raw['y'], pointcloud_raw['z']],dtype='float64').T)

        # Load object labels from pickle data
        with open(frame['object_labels_path'], 'rb') as pkl_file:
            objects_labels = pickle.load(pkl_file)

        bbox_list_3d = []
        bbox_list_2d = []
        kitti_labels = []
        pd_data = pd.DataFrame(pointcloud_raw)
        object_idxs = pd_data.id.value_counts()
        for label in objects_labels:
            # 筛选有效的物体lable
            if label.carla_id not in object_idxs and label.label_type != "vehicle":
                continue
            # Kitti Object - Type
            if label.label_type == 'vehicle':
                label_type = 'Car'
            elif label.label_type == 'pedestrian':
                label_type = 'Pedestrian'
            elif label.label_type == 'cyclist':
                label_type = 'Cyclist'
            else:
                label_type = 'DontCare'

            if not is_valid_distance(lidar_trans.location, label.transform.location):
                continue
            
            # 将点云数据的center转到物体中心
            if label.label_type != "pedestrian":
                label.transform.location.z += label.bounding_box.extent.z

            # Convert object label to open3d bbox type in lidar coordinate
            o3d_bbox = bbox_to_o3d_bbox_in_target_coordinate(label, lidar_trans)

            # Check lidar points in bbox
            # occlusion = cal_occlusion(o3d_pcd, o3d_bbox, label_type)
            occlusion = cal_occlusion_by_volumn(o3d_pcd, o3d_bbox, label_type)
            if occlusion < 0:
                continue

            # cal 2d bbox
            # 筛选出3d包围盒的点云
            maskpoints = o3d_bbox.get_point_indices_within_bounding_box(o3d.cpu.pybind.utility.Vector3dVector(pointcloud_with_id_label[:, :3]))
            # 筛选地面点
            maskpoints = [i for i in maskpoints if int(pointcloud_with_id_label[i, 3]) != 98 or int(pointcloud_with_id_label[i, 4]) == 10]

            # Note: 这里用两种方法计算2d包围盒，一种是以实际3d点云计算2d包围盒，一种是以3d包围盒的顶点计算2d包围盒
            # 以实际3d点云计算2d包围盒
            bbox_2d_filtered, truncated_flag = cal_bbox_2d_by_point(cam_mat, T_lc, image.shape, pointcloud_with_id_label[maskpoints, :3])
            # 如果bbox_2d_filtered没有点在图像内，则跳过
            if len(bbox_2d_filtered) == 0:
                continue
            # 以3d包围盒的点云计算2d包围盒
            bbox_2d_untruncated = cal_bbox_2d_by_vertex(o3d_bbox, cam_mat, T_lc)

            # 如果bbox有点在图像外，则用筛选后的点
            if truncated_flag:
                bbox_2d = bbox_2d_filtered
            # 否则，用bbox 8个顶点画框
            else:
                bbox_2d = bbox_2d_untruncated

            # # For Debug
            # # Draw 2d bbox
            # plt.imshow(image)
            # plt.plot([bbox_2d[0], bbox_2d[2]], [bbox_2d[1], bbox_2d[1]], 'r')
            # cv2.rectangle(image, (bbox_2d[0], bbox_2d[1]), (bbox_2d[2], bbox_2d[3]), (1, 0, 0), 2)
            # # save
            # cv2.imwrite(f"bbox_{index}.png", image)

            truncated = cal_truncated(image.shape[0], image.shape[1], bbox_2d_untruncated)

            # Ignore backward vehicles
            if o3d_bbox.center[0] < 0:
                truncated = 1.0

            o3d_bbox = bbox_to_o3d_bbox_in_target_coordinate(label, cam_trans)
            # print(f"o3d_bbox in camera coordinate: {o3d_bbox}")

            rotation_y = -math.radians(label.transform.rotation.yaw - cam_trans.rotation.yaw)
            rotation_y = math.atan2(math.sin(rotation_y), math.cos(rotation_y))

            bbox_center = np.asarray(o3d_bbox.center)
            theta = math.atan2(-bbox_center[0], bbox_center[2])
            alpha = rotation_y - theta
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))

            kitti_label = generate_kitti_labels(label_type, truncated, occlusion, alpha,
                                                bbox_2d, o3d_bbox, rotation_y)

            kitti_labels.append(kitti_label)
            bbox_list_3d.append(o3d_bbox)
            bbox_list_2d.append(bbox_2d)

        # Preview each frame label result
        # if index < 35:
        #     return
        # o3d_pcd.rotate(T_lc[0:3, 0:3], np.array([0, 0, 0]))
        # o3d_pcd.translate(T_lc[0:3, 3])
        # cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        # preview_obj = [o3d_pcd, cam_coord]
        # if len(bbox_list_3d) <= 0:
        #     return
        # for bbox3d in bbox_list_3d:
        #     center = np.asarray(bbox3d.center)
        #     box_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=center)
        #     box_coord.rotate(bbox3d.R)
        #     preview_obj.append(box_coord)
        #     preview_obj.append(bbox3d)
        #     print(f"bbox3d: {bbox3d}")
        # print(frame_id)
        # o3d.visualization.draw_geometries(preview_obj)

        # Output dataset in kitti format
        output_dir = os.path.join(self.output_dir, "training")

        write_calib(output_dir, cache_id, lidar_trans, cam_trans, cam_mat)
        write_label(output_dir, cache_id, kitti_labels)
        write_image(output_dir, cache_id, image)
        write_pointcloud(output_dir, cache_id, pointcloud_raw)
        write_status(output_dir, cache_id, timestamp, lidar_trans, vehicle_status)


def main():
    argparser = argparse.ArgumentParser(description="Select fragments of raw dataset to kitti format")
    argparser.add_argument('--root_dir', default=RAW_DATA_PATH,help='Root dir of raw dataset')
    argparser.add_argument('--output_root_dir', '-o',default='record',help='Output dir in dataset folder')
    argparser.add_argument('--output_name', '-n',default='kitti_object',help='Output name')
    argparser.add_argument('--record', '-r',default='record_2025_0108_2211',help='Rawdata Record ID. e.g. record_2022_0113_1337')
    argparser.add_argument('--vehicle', '-v',default='all',help='Vehicle name. e.g. `vehicle.tesla.model3_1`. Default to all vehicles. ')
    argparser.add_argument('--lidar', '-l',default='velodyne_semantic',help='Lidar name. e.g. sensor.lidar.ray_cast_4')
    argparser.add_argument('--camera', '-c',default='image_2',help='Camera name. e.g. sensor.camera.rgb_2')
    argparser.add_argument("--clip_length", default=80, type=int, help="The length of selected fragments")
    argparser.add_argument("--train", default=True, type=bool, help="Training or Testing dataset")

    args = argparser.parse_args()


    record_name = args.record
    if args.vehicle == 'all':
        vehicle_name_list = [os.path.basename(x) for x in glob.glob('{}/{}/vehicle.*'.format(RAW_DATA_PATH, record_name))]
    else:
        vehicle_name_list = [args.vehicle]

    for vehicle_name in vehicle_name_list:
        print("Process {} - {}".format(record_name, vehicle_name))
        world_dir = os.path.join(args.root_dir, record_name, "others.world_0")
        selected_frame_list = select_fragments(world_dir, args.clip_length)
        
        print(f"Gather rawdata to dataframe ...")
        if args.output_root_dir == '':
            output_dir = f"{DATASET_PATH}/{args.record_name}/{vehicle_name}/{args.output_name}"
        else:
            output_dir = f"{DATASET_PATH}/{args.output_root_dir}/{vehicle_name}/{args.output_name}"
        cache_index = len(os.listdir(os.path.join(output_dir, "training", "calib"))) if os.path.exists(os.path.join(output_dir, "training", "calib")) else 0
        print(f"Output dir: {output_dir}")
        print(f"Cache index: {cache_index}")
        rawdata_df = gather_rawdata_to_dataframe(record_name,
                                                vehicle_name,
                                                args.lidar,
                                                args.camera,
                                                cache_index,
                                                selected_frame_list)
        kitti_obj_label_tool = KittiObjectLabelTool(rawdata_df, output_dir, args.clip_length)
        print(f"Processing ...")
        kitti_obj_label_tool.process()

if __name__ == "__main__":
    main()    


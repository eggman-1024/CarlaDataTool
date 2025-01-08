#!/usr/bin/python3
import argparse
from ast import Delete
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

import numpy as np

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import RAW_DATA_PATH, DATASET_PATH
from label_tools.kitti_object.kitti_object_data_loader import *
from label_tools.kitti_object.kitti_object_helper import *


def gather_rawdata_to_dataframe(record_name: str, vehicle_name: str, lidar_path: str, camera_path: str):
    '''
    Gather rawdata to a pandas dataframe
    :param record_name: str, record name
    :param vehicle_name: str, vehicle name
    :param lidar_path: str, lidar path
    :param camera_path: str, camera path
    :return: pd.DataFrame with columns ['index', 'frame', 'timestamp', 'vehicle_pose', 'vehicle_status', 
                                        'object_labels_path', 'lidar_rawdata_path', 'lidar_pose', 
                                        'camera_rawdata_path', 'camera_pose', 'camera_matrix']
    '''
    rawdata_frames_df = pd.DataFrame()
    vehicle_poses_df = load_vehicle_pose("{}/{}/{}".format(RAW_DATA_PATH, record_name, vehicle_name))
    rawdata_frames_df = vehicle_poses_df
    rawdata_frames_df = rawdata_frames_df.reset_index(drop=False)

    object_labels_path_df = load_object_labels("{}/{}/others.world_1".format(RAW_DATA_PATH, record_name))
    rawdata_frames_df = pd.merge(rawdata_frames_df, object_labels_path_df, how='outer', on='frame')

    lidar_rawdata_df = load_lidar_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}/{lidar_path}")
    rawdata_frames_df = pd.merge(rawdata_frames_df, lidar_rawdata_df, how='outer', on='frame')

    camera_rawdata_path_df = load_camera_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}/{camera_path}")
    rawdata_frames_df = pd.merge(rawdata_frames_df, camera_rawdata_path_df, how='outer', on='frame')

    return rawdata_frames_df


def generate_image_sets(path_to_kitti_object: str, train_ratio=0.55):
    calib_data_list = sorted(glob.glob(f"{path_to_kitti_object}/training/calib/*.txt"))
    train_sets = []
    val_sets = []
    for i, item in enumerate(calib_data_list):
        if i < len(calib_data_list) * train_ratio:
            frame_id = os.path.splitext(os.path.split(item)[-1])[0]
            train_sets.append(f"{frame_id}\n")
        else:
            frame_id = os.path.splitext(os.path.split(item)[-1])[0]
            val_sets.append(f"{frame_id}\n")
    train_val_sets = train_sets + val_sets
       
    image_sets_dir = f"{path_to_kitti_object}/ImageSets"
    os.makedirs(image_sets_dir, exist_ok=True)
    with open(f"{image_sets_dir}/train.txt", 'w') as file:
        file.writelines(train_sets)
    with open(f"{image_sets_dir}/val.txt", 'w') as file:
        file.writelines(val_sets)
    # with open(f"{image_sets_dir}/test.txt", 'w') as file:
    #     file.writelines(image_sets)
    with open(f"{image_sets_dir}/trainval.txt", 'w') as file:
        file.writelines(train_val_sets)


class KittiObjectLabelTool:
    def __init__(self, record_name, vehicle_name, rawdata_df: pd.DataFrame, output_dir=None):
        self.record_name = record_name
        self.vehicle_name = vehicle_name
        self.rawdata_df = rawdata_df[rawdata_df.camera_pose.notna()]
        self.range_max = 120.0
        self.range_min = 1.0
        self.points_min = 10
        self.output_dir = output_dir

    def process(self):
        debug = False #TODO:!!
        if not debug:
            start = time.time()
            thread_pool = ThreadPool()
            thread_pool.starmap(self.process_frame, self.rawdata_df.iterrows())
            thread_pool.close()
            thread_pool.join()
            if self.output_dir == '':
                output_dir = f"{DATASET_PATH}/{self.record_name}/{self.vehicle_name}/kitti_object"
            else:
                output_dir = f"{DATASET_PATH}/{self.output_dir}/kitti_object"
            generate_image_sets(output_dir)
            print("Cost: {:0<3f}s".format(time.time() - start))
        else:
            start = time.time()
            for index, frame in self.rawdata_df.iterrows():
                index = 1120
                frame = self.rawdata_df.iloc[index]
                self.process_frame(index, frame)
            if self.output_dir == '':
                output_dir = f"{DATASET_PATH}/{self.record_name}/{self.vehicle_name}/kitti_object"
            else:
                output_dir = f"{DATASET_PATH}/{self.output_dir}/kitti_object"
            generate_image_sets(output_dir)
            print("Cost: {:0<3f}s".format(time.time()-start))

        shutil.rmtree(f"{output_dir}/testing",ignore_errors=True)
        # shutil.copytree(f"{output_dir}/training", f"{output_dir}/testing")


    def process_frame(self, index, frame):
        # print(f"frame: {frame}")
        index = "{:0>6d}".format(index)
        frame_id = "{:0>6d}".format(frame['frame'])
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
        if self.output_dir == '':
            output_dir = f"{DATASET_PATH}/{self.record_name}/{self.vehicle_name}/kitti_object/training"
        else:
            output_dir = f"{DATASET_PATH}/{self.output_dir}/kitti_object/training"
        write_calib(output_dir, index, lidar_trans, cam_trans, cam_mat)
        write_label(output_dir, index, kitti_labels)
        write_image(output_dir, index, image)
        write_pointcloud(output_dir, index, pointcloud_raw)
        write_status(output_dir, index, timestamp, lidar_trans, vehicle_status)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record', '-r',
        nargs='+',
        # default=['record_2025_0106_2019', 'record_2025_0106_2025', 'record_2025_0106_2032', 'record_2025_0106_2038', 'record_2025_0106_2045', 'record_2025_0106_2051'],  # 默认值为一个列表
        default=['record_2025_0106_2019', 'record_2025_0106_2025'],  # 默认值为一个列表
        help='Rawdata Record ID. e.g. record_2022_0113_1337'
    )
    argparser.add_argument(
        '--vehicle', '-v',
        default='all',
        help='Vehicle name. e.g. `vehicle.tesla.model3_1`. Default to all vehicles. '
    )
    argparser.add_argument(
        '--lidar', '-l',
        default='velodyne_semantic',
        help='Lidar name. e.g. sensor.lidar.ray_cast_4'
    )
    argparser.add_argument(
        '--camera', '-c',
        default='image_2',
        help='Camera name. e.g. sensor.camera.rgb_2'
    )
    argparser.add_argument(
        '--output_dir', '-o',
        default='',
        help='Output dir in dataset folder'
    )

    args = argparser.parse_args()

    record_name_list = args.record
    for record_name in record_name_list:
        if args.vehicle == 'all':
            vehicle_name_list = [os.path.basename(x) for x in glob.glob('{}/{}/vehicle.*'.format(RAW_DATA_PATH, record_name))]
        else:
            vehicle_name_list = [args.vehicle]

        for vehicle_name in vehicle_name_list:
            rawdata_df = gather_rawdata_to_dataframe(record_name,
                                                    vehicle_name,
                                                    args.lidar,
                                                    args.camera)
            print("Process {} - {}".format(record_name, vehicle_name))
            kitti_obj_label_tool = KittiObjectLabelTool(record_name, vehicle_name, rawdata_df, args.output_dir)
            kitti_obj_label_tool.process()
    


if __name__ == '__main__':
    main()

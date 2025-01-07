#!/usr/bin/python3
import copy
import numpy as np
import open3d as o3d
import cv2
import os
import sys
from pathlib import Path
import transforms3d.euler

sys.path.append(Path(__file__).parent.parent.as_posix())
from utils.geometry_types import Transform, Location
from utils.label_types import ObjectLabel
from utils.transform import bbox_to_o3d_bbox


class Param:
    # occlusion_params = {'Car':{'POINTS_MIN': 10, 'THRESHOLD_1': 250, 'THRESHOLD_2': 125},
    #                     'Pedestrian':{'POINTS_MIN': 8, 'THRESHOLD_1': 150, 'THRESHOLD_2': 80},
    #                     'Cyclist':{'POINTS_MIN': 6, 'THRESHOLD_1': 120, 'THRESHOLD_2': 60}}
    occlusion_params = {'Car':{'IOU_MIN': 0.1, 'IOU_THRESHOLD_1': 0.7, 'IOU_THRESHOLD_2': 0.35, 'DISTANCE_THRESHOLD_1': 20, 'DISTANCE_THRESHOLD_2': 80},
                        'Pedestrian':{'IOU_MIN': 0.1, 'IOU_THRESHOLD_1': 0.4, 'IOU_THRESHOLD_2': 0.2, 'DISTANCE_THRESHOLD_1': 20, 'DISTANCE_THRESHOLD_2': 80},
                        'Cyclist':{'IOU_MIN': 0.1, 'IOU_THRESHOLD_1': 0.6, 'IOU_THRESHOLD_2': 0.3, 'DISTANCE_THRESHOLD_1': 20, 'DISTANCE_THRESHOLD_2': 80}}
    RANGE_MIN = 1.0
    RANGE_MAX = 120.0


def transform_lidar_point_to_cam(point, T_lc: np.array):
    p = np.append(point, [1.0])
    p_c = np.matmul(T_lc, p)
    return p_c


def project_point_to_image(point_in_cam,
                           cam_mat: np.array):
    p_c = point_in_cam
    p_c = p_c[0:3] / p_c[2]
    p_uv = np.matmul(cam_mat, p_c)
    p_uv = p_uv[0:2].astype(int)
    return p_uv


def transform_o3d_bbox(o3d_bbox: o3d.geometry.OrientedBoundingBox, transform_mat: np.array):
    o3d_bbox.rotate(transform_mat[0:3, 0:3], np.array([0, 0, 0]))
    o3d_bbox.translate(transform_mat[0:3, 3])
    return o3d_bbox


def object_label_to_target_coordinate(pc: np.array, target_transform: Transform):
    '''
    Transform the point cloud and label to the target coordinate.
    :param pc: np.array, the point cloud to be transformed
    :param target_transform: Transform, the target transform
    :return: np.array, the transformed point cloud
    '''
    pc = np.append(pc, np.ones((pc.shape[0], 1)), axis=1)
    world_to_target = target_transform.get_inverse_matrix()
    pc_in_target = np.matmul(world_to_target, pc.T).T[:, 0:3]
    return pc_in_target


def bbox_to_o3d_bbox_in_target_coordinate(label: ObjectLabel, target_transform: Transform):
    '''
    Transform the label bounding box to the open3d format in target coordinate.
    :param label: ObjectLabel, the label to be transformed
    :param target_transform: Transform, the target transform
    :return: o3d.geometry.OrientedBoundingBox, the transformed bounding box
    '''
    world_to_target = target_transform.get_inverse_matrix()
    o3d_bbox = bbox_to_o3d_bbox(label, label.bounding_box) 
    o3d_bbox = transform_o3d_bbox(o3d_bbox, world_to_target)
    o3d_bbox.color = np.array([1.0, 0, 0])
    return o3d_bbox



def o3d_bbox_rotation_to_rpy(o3d_bbox: o3d.geometry.OrientedBoundingBox):
    # o3d bbox R to euler RPY in radian
    roll, pitch, yaw = transforms3d.euler.mat2euler(o3d_bbox.R)
    return roll, pitch, yaw


def cal_truncated(image_length, image_width, bbox_2d: list) -> float:
    """
    Calculate the truncation ratio of a bounding box in an image.
    The truncation ratio is a measure of how much of the object is outside the image boundaries.
    It ranges from 0 (non-truncated) to 1 (fully truncated).
    Args:
        image_length (int): The height of the image.
        image_width (int): The width of the image.
        bbox_2d (list): A list of four floats representing the bounding box coordinates [x_min, y_min, x_max, y_max].
    Returns:
        float: The truncation ratio, where 0 means the object is fully within the image boundaries and 1 means the object is fully outside the image boundaries.
    """
    # Calculate truncated.
    # from 0 (non-truncated) to 1 (truncated), where
    # truncated refers to the object leaving image boundaries

    # [x_min y_min x_max y_max]
    bbox_2d_in_img = copy.deepcopy(bbox_2d)
    bbox_2d_in_img[0] = max(bbox_2d[0], 0)
    bbox_2d_in_img[1] = max(bbox_2d[1], 0)
    bbox_2d_in_img[2] = min(bbox_2d[2], image_width)
    bbox_2d_in_img[3] = min(bbox_2d[3], image_length)

    size1 = float((bbox_2d_in_img[2] - bbox_2d_in_img[0])) * float((bbox_2d_in_img[3] - bbox_2d_in_img[1]))
    size2 = float((bbox_2d[2] - bbox_2d[0])) * float((bbox_2d[3] - bbox_2d[1]))
    if size2 == 0 or size1 == 0:
        return 1.0
    truncated = size1 / size2
    truncated = max(truncated, 0.0)
    truncated = min(truncated, 1.0)
    truncated = 1.0 - truncated
    return truncated

def cal_occlusion(pcd: o3d.geometry.PointCloud, bbox_3d: o3d.geometry.OrientedBoundingBox, type: str):
    occlusion = 2
    p_in_bbox = bbox_3d.get_point_indices_within_bounding_box(pcd.points)
    p_num = len(p_in_bbox)
    if p_num < Param.occlusion_params[type]['POINTS_MIN']:
        return -1
    else:
        occlusion = 0
    if p_num + bbox_3d.center[0] < Param.occlusion_params[type]['THRESHOLD_1']:
        occlusion = 1
    if p_num + bbox_3d.center[0] < Param.occlusion_params[type]['THRESHOLD_2']:
        occlusion = 2
    return occlusion

# 遮挡程度计算流程
# 1. 筛选出所有在lidar范围内的3d bbox
# 2. 投影到Image上，得到2d bbox(需要裁剪)
# 3. 找到与目标物体的2d bbox相交的所有物体的距离、2d bbox
# 4. 只保留距离小于当前目标物体的2d bbox
# 5. 计算目标物体的2d bbox与其他物体的2d bbox的重叠面积
def cal_occlusion_by_image(bbox_2d: list, bbox_3d: o3d.geometry.OrientedBoundingBox, labels: list):
    pass


def cal_occlusion_by_volumn(pcd: o3d.geometry.PointCloud, bbox_3d: o3d.geometry.OrientedBoundingBox, type: str):

    def distance_correction_factor(distance, min_distance, max_distance):
        # 当距离小于min_distance时，修正因子大于1，表示补偿
        # 当距离大于max_distance时，修正因子小于1，表示距离对遮挡程度的影响逐渐减弱
        if distance < min_distance:
            return 1.0 + (min_distance - distance) / min_distance
        elif distance > max_distance:
            return 1.0 - (distance - max_distance) / max_distance
        else:
            return 1.0

    # 以点云的最小外接矩形为准
    p_in_bbox_indice = bbox_3d.get_point_indices_within_bounding_box(pcd.points)
    # 如果点数小于4，说明点云不够，无法计算遮挡程度
    if len(p_in_bbox_indice) < 4:
        return -1
    p_in_bbox = pcd.select_by_index(p_in_bbox_indice)
    
    # inner_bbox_3d = p_in_bbox.get_oriented_bounding_box()
    inner_bbox_3d = p_in_bbox.get_axis_aligned_bounding_box()
    inner_volume = inner_bbox_3d.volume()
    # inner_volume = inner_bbox_3d.extent[0] * inner_bbox_3d.extent[1] * inner_bbox_3d.extent[2]
    outter_volume = bbox_3d.extent[0] * bbox_3d.extent[1] * bbox_3d.extent[2]
    # 计算重叠体积
    iou_volume = inner_volume / outter_volume


    distance = bbox_3d.center[0]
    distance_factor = distance_correction_factor(distance, Param.occlusion_params[type]['DISTANCE_THRESHOLD_1'], Param.occlusion_params[type]['DISTANCE_THRESHOLD_2'])
    iou_volume = iou_volume * distance_factor

    if iou_volume < Param.occlusion_params[type]['IOU_MIN']:
        return -1
    if iou_volume > Param.occlusion_params[type]['IOU_THRESHOLD_1']:
        occlusion = 0
    elif iou_volume > Param.occlusion_params[type]['IOU_THRESHOLD_2']:
        occlusion = 1
    else:
        occlusion = 2
    return occlusion




def cal_bbox_2d_by_point(cam_mat: np.array, T_lc: np.array, image_size: tuple, maskpoints=None):
    image_width = image_size[1]
    image_height = image_size[0]

    bbox_points_2d_x = []
    bbox_points_2d_y = []
    for p in maskpoints:
        p_c = transform_lidar_point_to_cam(p, T_lc)
        p_uv = project_point_to_image(p_c, cam_mat)

        bbox_points_2d_x.append(p_uv[0])
        bbox_points_2d_y.append(p_uv[1])

    filtered_bbox_points_2d_x = []
    filtered_bbox_points_2d_y = []
    truncated_flag = False
    for x, y in zip(bbox_points_2d_x, bbox_points_2d_y):
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            truncated_flag = True
            continue
        filtered_bbox_points_2d_x.append(x)
        filtered_bbox_points_2d_y.append(y)
    # 如果bbox全部在图像外，则返回bbox_2d和空的bbox_2d_filtered
    if len(filtered_bbox_points_2d_x) == 0 or len(filtered_bbox_points_2d_y) == 0:
        return [], True
    
    bbox_2d_filtered = [min(filtered_bbox_points_2d_x), min(filtered_bbox_points_2d_y), max(filtered_bbox_points_2d_x), max(filtered_bbox_points_2d_y)]
    return bbox_2d_filtered, truncated_flag


def cal_bbox_2d_by_vertex(o3d_bbox: o3d.geometry.OrientedBoundingBox, cam_mat: np.array, T_lc: np.array):
    vertex_bbox_points_2d_x = []
    vertex_bbox_points_2d_y = []
    vertex_points = np.asarray(o3d_bbox.get_box_points())
    for p in vertex_points:
        p_c = transform_lidar_point_to_cam(p, T_lc)
        p_uv = project_point_to_image(p_c, cam_mat)

        vertex_bbox_points_2d_x.append(p_uv[0])
        vertex_bbox_points_2d_y.append(p_uv[1])
    bbox_2d = [min(vertex_bbox_points_2d_x), min(vertex_bbox_points_2d_y), max(vertex_bbox_points_2d_x), max(vertex_bbox_points_2d_y)]

    return bbox_2d


def points_to_image(points, Tr_velo_cam, R0, P):
    """
    将3D点投影到2D图像上
    :param points: 3D点 (N, 3)
    :param Tr_velo_cam: 从雷达坐标系到相机坐标系的变换矩阵 (3, 4)
    :param R0: 从相机坐标系到图像坐标系的变换矩阵 (3, 3)
    :param P: 投影矩阵 (3, 4)
    :return: 2D点 (N, 2)
    """
    pts_3d_velo = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
    pts_3d_ref = pts_3d_velo @ Tr_velo_cam.T  # (N, 3)
    pts_3d_rect = (R0 @ pts_3d_ref.T).T  # (N, 3)
    pts_3d_rect = np.hstack([pts_3d_rect, np.ones((pts_3d_rect.shape[0], 1))])  # (N, 4)
    pts_2d = pts_3d_rect @ P.T  # (N, 3)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def is_valid_distance(source_location: Location, target_location: Location):
    dist = np.linalg.norm(source_location.get_vector() - target_location.get_vector())
    if Param.RANGE_MIN < dist < Param.RANGE_MAX:
        return True
    else:
        return False

def convert_v_to_lidar_coordinate(point, lidar_trans):
    # 将velodyne_files中的点云的v_x,v_y从世界坐标系转为自车雷达坐标系
    # 构建3×3旋转矩阵
    R = lidar_trans.rotation.get_rotation_matrix()
    # 旋转矩阵的逆矩阵
    R_inv = np.linalg.inv(R)
    # 旋转矩阵的逆矩阵的转置
    R_inv_T = R_inv.T

    # 读取点云数据中的vx,vy,vz
    v = np.stack([point['vx'], point['vy'], point['vz']], axis=-1)
    v_ego = v @ R_inv_T
    # print(f"v_ego:{v_ego[:5]}")
    point['vx'] = v_ego[:,0]
    point['vy'] = v_ego[:,1]
    point['vz'] = v_ego[:,2]
    return point


def write_pointcloud(output_dir: str, frame_id: str, lidar_data: np.array):
    lidar_dir = f"{output_dir}/velodyne"
    # 暂时删除非必要数据
    # lidar_data = np.array([lidar_data['x'],lidar_data['y'],lidar_data['z'],lidar_data['v'],lidar_data['vx'],lidar_data['vy'],lidar_data['v_cps'],lidar_data['intensity']]).T
    # lidar_data = np.array(lidar_data.tolist())
    os.makedirs(lidar_dir, exist_ok=True)
    file_path = "{}/{}.bin".format(lidar_dir, frame_id)
    lidar_data.tofile(file_path)


def write_image(output_dir: str, frame_id: str, image: np.array):
    image_dir = f"{output_dir}/image_2"
    os.makedirs(image_dir, exist_ok=True)
    file_path = "{}/{}.png".format(image_dir, frame_id)
    cv2.imwrite(file_path, image)


def write_label(output_dir, frame_id, kitti_labels):
    label_dir = f"{output_dir}/label_2"
    os.makedirs(label_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(label_dir, frame_id)

    if len(kitti_labels) < 1:
        kitti_labels.append('DontCare -1 -1 -10 0 1 0 1 -1 -1 -1 -1000 -1000 -1000 -10\n')

    with open(file_path, 'w') as label_file:
        label_file.writelines(kitti_labels)

# 记录自车速度、加速度与lidar位姿(x,y,z,pitch,yaw,roll)
def write_status(output_dir, frame_id, timestamp, lidar_trans, vehicle_status):
    status_dir = f"{output_dir}/status"
    os.makedirs(status_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(status_dir, frame_id)
    lidar_x = lidar_trans.location.x
    lidar_y = lidar_trans.location.y
    lidar_z = lidar_trans.location.z
    lidar_roll = lidar_trans.rotation.roll
    lidar_pitch = lidar_trans.rotation.pitch
    lidar_yaw = lidar_trans.rotation.yaw


    with open(file_path, 'w') as status_file:
        status_file.writelines(f"timestamp: {timestamp}\n")
        status_file.writelines(f"lidar: {lidar_x} {lidar_y} {lidar_z} {lidar_roll} {lidar_pitch} {lidar_yaw}\n")
        status_file.writelines(f"vehicle: {vehicle_status[0]} {vehicle_status[1]} {vehicle_status[2]} {vehicle_status[3]} {vehicle_status[4]} {vehicle_status[5]}\n")

def write_calib(output_dir, frame_id, lidar_trans: Transform, cam_trans: Transform, camera_mat: np.array):
    """ Saves the calibration matrices to a file.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame.
    """
    calib_dir = f"{output_dir}/calib"
    os.makedirs(calib_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(calib_dir, frame_id)

    camera_mat = np.concatenate((camera_mat, np.array([[0.0], [0.0], [0.0]])), axis=1)
    camera_mat = camera_mat.reshape(1, 12)
    camera_mat_str = ""
    for x in camera_mat[0]:
        camera_mat_str += ' '
        camera_mat_str += str(x)
    camera_mat_str += '\n'
    camera_mat_str = camera_mat_str[1:]

    calib_str = list()
    calib_str.append(f"P0: {camera_mat_str}")
    calib_str.append(f"P1: {camera_mat_str}")
    calib_str.append(f"P2: {camera_mat_str}")
    calib_str.append(f"P3: {camera_mat_str}")

    calib_str.append("R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n")

    velo_to_cam = np.matmul(cam_trans.get_inverse_matrix(), lidar_trans.get_matrix())
    velo_to_cam = velo_to_cam[0:3, :]
    velo_to_cam = velo_to_cam.reshape(1, 12).tolist()
    velo_to_cam_str = "Tr_velo_to_cam: "
    for x in velo_to_cam[0]:
        velo_to_cam_str += str(x)
        velo_to_cam_str += ' '
    velo_to_cam_str += '\n'

    # Note: 由于计算出来的Tr_velo_to_cam存在浮点数误差以及精度问题，因此直接写入固定值
    calib_str.append("Tr_velo_to_cam: 0.0 -1.0 0.0 0.0 0.0 0.0 -1.0 -0.4 1.0 0.0 0.0 -2.0\n")
    calib_str.append("Tr_imu_to_velo: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")

    with open(file_path, 'w') as calib_file:
        calib_file.writelines(calib_str)
        calib_file.close()


def generate_kitti_labels(label_type: str,
                          truncated: float,
                          occlusion: float,
                          alpha: float,
                          bbox_2d: list,
                          bbox_3d: o3d.geometry.OrientedBoundingBox,
                          rotation_y: float):
    # Note: Kitti Object 3d bbox location is top-plane-center, not the bbox center
    label_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(label_type, truncated, occlusion, alpha,
                                                                         bbox_2d[0], bbox_2d[1],
                                                                         bbox_2d[2], bbox_2d[3],
                                                                         bbox_3d.extent[2],
                                                                         bbox_3d.extent[1],
                                                                         bbox_3d.extent[0],
                                                                         bbox_3d.center[0],
                                                                         bbox_3d.center[1] + (bbox_3d.extent[2] / 2.0),
                                                                         bbox_3d.center[2],
                                                                         rotation_y)
    return label_str

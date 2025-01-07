import copy
import numba
import numpy as np
import random
import torch

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

def extend_matrix(r0_rect, tr_velo_to_cam, P2):
    '''
    r0_rect: shape=(3, 3)
    tr_velo_to_cam: shape=(3, 4)
    P2: shape=(3, 4)
    return: shape=(4, 4)
    '''
    R0_rect_extend = np.eye(4, dtype=r0_rect.dtype)
    R0_rect_extend[:3, :3] = r0_rect
    R0_rect = R0_rect_extend

    Tr_velo_to_cam = np.concatenate([tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)

    P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)

    return R0_rect, Tr_velo_to_cam, P2



# Modified from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L609
def remove_outside_points(points, r0_rect, tr_velo_to_cam, P2, image_shape):
    """Remove points which are outside of image.
    Args:
        points (np.ndarray, shape=[N, 3+dims]): Total points.
        rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.
        image_shape (list[int]): Shape of image.
    Returns:
        np.ndarray, shape=[N, 3+dims]: Filtered points.
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = points_camera2lidar(frustum.T[None, ...], tr_velo_to_cam, r0_rect) # (1, 8, 3)
    group_rectangle_vertexs_v = group_rectangle_vertexs(frustum)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, 1)
    points = points[indices.reshape([-1])]
    return points

# Copied from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L609
def projection_matrix_to_CRT_kitti(proj):
    """Split projection matrix of kitti.
    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.
    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.
    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T

# Copied from https://github.com/open-mmlab/mmdetection3d/blob/f45977008a52baaf97640a0e9b2bbe5ea1c4be34/mmdet3d/core/bbox/box_np_ops.py#L661
def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """Get frustum corners in camera coordinates.
    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float, optional): Nearest distance of frustum.
            Defaults to 0.001.
        far_clip (float, optional): Farthest distance of frustum.
            Defaults to 100.
    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
        dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def points_camera2lidar(points, tr_velo_to_cam, r0_rect):
    '''
    points: shape=(N, 8, 3) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 8, 3)
    '''
    extended_xyz = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    xyz = extended_xyz @ rt_mat.T
    return xyz[..., :3]


def group_rectangle_vertexs(bboxes_corners):
    '''
    将8个角点按照顺序分为6个矩形面
    bboxes_corners: shape=(n, 8, 3)
            ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    return: shape=(n, 6, 4, 3)
    '''
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]], axis=1) # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]], axis=1) # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]], axis=1) # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]], axis=1) # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs


def group_plane_equation(bbox_group_rectangle_vertexs):
    '''
    输入6个矩形面的4个角点坐标，返回6个平面方程的参数
    bbox_group_rectangle_vertexs: shape=(n, 6, 4, 3)
    return: shape=(n, 6, 4)
    '''
    # 1. generate vectors for a x b
    vectors = bbox_group_rectangle_vertexs[:, :, :2] - bbox_group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1]) # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', bbox_group_rectangle_vertexs[:, :, 0], normal_vectors) # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params


@numba.jit(nopython=True)
def points_in_bboxes(points, plane_equation_params):
    '''
    这段代码的作用是判断点云中的每个点是否位于多个3D边界框（bbox）内部。
    它通过将每个点与边界框的平面方程进行比较，
    确定点是否在该边界框的所有平面之内
    points: shape=(N, 3)
    plane_equation_params: shape=(n, 6, 4)
    return: shape=(N, n), bool
    '''
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks
import os
import numpy as np

origin_path = '/home/public/zrtdata/carla4d_kitti/training/label_2'
rewrite_path = '/home/public/mzydata/label_2_fix'
os.makedirs(rewrite_path, exist_ok=True)

fix_hwl_dict = {
    '1.4872888326644897' : ['1.600', '0.8592574596405029', '1.4872888326644897'],
    '1.6428436040878296' : ['1.800', '0.600', '1.6428436040878296'],
    '1.8046669960021973' : ['1.620', '0.85568767786026', '1.8046669960021973'],
    '1.8354405164718628' : ['1.800', '0.850', '1.8354405164718628'],
    '2.0333523750305176' : ['1.550', '0.8025798797607422', '2.0333523750305176'],
    '2.2094459533691406' : ['1.500', '0.8670341968536377', '2.2094459533691406'],
    '2.3557403087615967' : ['1.550', '0.7636788487434387', '2.3557403087615967']
}

def compute_box3d_corner(h, w, l, x, y, z, yaw):
    """
    根据kitti格式的3d框参数计算3d bbox的8个角点坐标
    输入：高度h, 宽度w, 长度l, 底部中心点x, y, z, 旋转角yaw
    返回：8个角点坐标 (3, 8)
    注意该3d框角点坐标位于相机坐标系下
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners += np.vstack([x, y, z])
    return corners


def kitti_box3d_to_box2d(h, w, l, x, y, z, yaw, R0, P2, img_size):
    """
    将kitti数据集格式的3d bbox参数转换为2d bbox参数
    输入：高度h, 宽度w, 长度l, 底部中心点x, y, z, 旋转角yaw, 相机内参R0, P2, 图像尺寸
    返回：左上角点和右下角点(x0, y0, x1, y1)
    """
    # 1.获取3d bbox的8个角点坐标（相机2坐标系下）
    corners_3d_cam2 = compute_box3d_corner(h, w, l, x, y, z, yaw)  # (3, 8)

    # 2.将相机坐标系的点转换到图像坐标系
    pts_3d_rect = corners_3d_cam2.T @ R0.T
    pts_3d_rect = np.hstack([pts_3d_rect, np.ones((pts_3d_rect.shape[0], 1))])
    pts_2d = pts_3d_rect @ P2.T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    points_2d = pts_2d[:, 0:2]  # (8, 2)

    # 3.根据这8个点计算2d bbox的左上角和右下角点，根据照片尺寸进行裁剪
    x0 = max(0, np.min(points_2d[:, 0]))
    y0 = max(0, np.min(points_2d[:, 1]))
    x1 = min(img_size[0], np.max(points_2d[:, 0]))
    y1 = min(img_size[1], np.max(points_2d[:, 1]))
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    return x0, y0, x1, y1

R0 = np.eye(3)
P2 = np.array([[640.0000000000001, 0.0, 640.0, 0.0],
                [0.0, 640.0000000000001, 360.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]])
img_size = (1280, 720)

# 重写文件中的Cyclist的hwl以修复数据
file_list = os.listdir(origin_path)
for file in file_list:
    file_path = os.path.join(origin_path, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if line[0] == 'Cyclist':
                h, w, l = line[8:11]
                h_fix, w_fix, l_fix = fix_hwl_dict[l]
                x, y, z, ry = line[11:15]
                x, y, z, ry = float(x), float(y), float(z), float(ry)
                x0, y0, x1, y1 = kitti_box3d_to_box2d(float(h_fix), float(w_fix), float(l_fix), x, y, z, ry, R0, P2, img_size)

                # 修复数据
                line[4] = str(x0)
                line[5] = str(y0)
                line[6] = str(x1)
                line[7] = str(y1)
                line[8] = h_fix
                line[9] = w_fix
                line[10] = l_fix
                line = ' '.join(line)
                with open(os.path.join(rewrite_path, file), 'a') as f:
                    f.write(line + '\n')
            else:
                with open(os.path.join(rewrite_path, file), 'a') as f:
                    f.write(' '.join(line) + '\n')

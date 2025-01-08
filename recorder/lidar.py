
import carla
import numpy as np
import struct

from recorder.sensor import Sensor


class Lidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Save as a Nx4 numpy array. Each row is a point (x, y, z, intensity)
        lidar_data = np.fromstring(bytes(sensor_data.raw_data),
                                   dtype=np.float32)
        lidar_data = np.reshape(
            lidar_data, (int(lidar_data.shape[0] / 4), 4))

        # Convert point cloud to right-hand coordinate system
        lidar_data[:, 1] *= -1

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy
        np.save("{}/{:0>10d}".format(save_dir,
                                     sensor_data.frame),
                lidar_data)
        return True


class SemanticLidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Save data as a Nx12 numpy array.
        lidar_data = np.fromstring(bytes(sensor_data.raw_data),
                                   dtype=np.dtype([
                                       ('x', np.float32),
                                       ('y', np.float32),
                                       ('z', np.float32),
                                       ('cos_angle', np.float32),
                                       ('v', np.float32),
                                       ('vx', np.float32),
                                       ('vy', np.float32),
                                       ('vz',np.float32),
                                       ('v_cps', np.float32),
                                       ('intensity', np.float32),
                                    #    ('norm_x', np.float32),
                                    #    ('norm_y', np.float32),
                                    #    ('norm_z', np.float32),
                                       ('id', np.uint32),
                                       ('label', np.uint32)
                                   ]))

        # Convert point cloud to right-hand coordinate system
        lidar_data['y'] *= -1
        lidar_data['vy'] *= -1
        # lidar_data['norm_y'] *= -1

        # 边录制数据边转换时间超过了每一帧时间间隔，会导致无法保存点云文件，所以先录制数据，后续手动转换
        # # 将vx, vy, vz转为自车雷达坐标系
        # # 读取此时lidar的transform
        # lidar_transform = self.carla_actor.get_transform()
        # R = lidar_transform.rotation.get_rotation_matrix()  # 构建3×3旋转矩阵
        # R_inv = np.linalg.inv(R) # 旋转矩阵的逆矩阵
        # R_inv_T = R_inv.T   # 旋转矩阵的逆矩阵的转置
        # # 将点云数据中的vx,vy,vz转为自车雷达坐标系
        # v = np.stack([lidar_data['vx'], lidar_data['vy'], lidar_data['vz']], axis=-1)
        # v_ego = v @ R_inv_T
        # lidar_data['vx'] = v_ego[:, 0]
        # lidar_data['vy'] = v_ego[:, 1]
        # lidar_data['vz'] = v_ego[:, 2]


        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy
        # 保存为.npy文件
        np.save("{}/{:0>10d}".format(save_dir,
                                     sensor_data.frame),
                lidar_data)
        
        # 构造文件名和路径
        # file_path = "{}/{:0>10d}.bin".format(save_dir, sensor_data.frame)

        # lidar_data.tofile(file_path)

        return True

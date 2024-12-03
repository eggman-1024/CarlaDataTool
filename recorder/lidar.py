
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

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy
        # 保存为.bin文件

        np.save("{}/{:0>10d}".format(save_dir,
                                     sensor_data.frame),
                lidar_data)
        
        # 构造文件名和路径
        # file_path = "{}/{:0>10d}.bin".format(save_dir, sensor_data.frame)

        # lidar_data.tofile(file_path)

        return True

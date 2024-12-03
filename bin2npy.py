import os
import numpy as np

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
    ])

def convert_bin_to_npy(in_directory, out_directory, dtype=dtype):
    for filename in os.listdir(in_directory):
        if filename.endswith(".bin"):
            bin_file_path = os.path.join(in_directory, filename)
            npy_file_path = os.path.join(out_directory, filename.replace(".bin", ".npy"))
            
            # 读取 .bin 文件
            lidar_data = np.fromfile(bin_file_path, dtype=dtype)
            
            # 保存为 .npy 文件
            np.save(npy_file_path, lidar_data)
            print(f"Converted {bin_file_path} to {npy_file_path}")

# 示例使用
bin_directory_path = r"raw_data\record_2024_0928_1819\vehicle.tesla.model3.master\velodyne_semantic"
npy_directory_path = r"raw_data\record_2024_0928_1819\vehicle.tesla.model3.master\velodyne"
convert_bin_to_npy(bin_directory_path, npy_directory_path)

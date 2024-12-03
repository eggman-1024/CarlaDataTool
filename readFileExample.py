import numpy as np
import pandas as pd

path = r"F:\Software\carla_tools\raw_data\record_2024_1025_0902\vehicle.tesla.model3.master\velodyne_semantic\0000000069.npy"
# path = r"F:\Software\carla_tools\dataset/record_2024_1025_0902/vehicle.tesla.model3.master/kitti_object/training/velodyne/000000.bin"


dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('cos_angle', np.float32), 
                  ('v', np.float32), ('vx', np.float32), ('vy', np.float32), ('vz', np.float32), ('v_cps', np.float32), 
                  ('intensity', np.float32), ('id', np.uint32), ('label', np.uint32)])

data = np.fromfile(path, dtype=np.float32).reshape(-1, 8)


# 读取npy文件
data = np.load(path).tolist()[:10]
print(data[:10])
# lidar_data = np.array([data['x'], data['y']]).T
lidar_data = np.array(data)
print(lidar_data)
print(f"type(lidar_data): {type(lidar_data)}")
# 保存为csv文件
# columns = ['x', 'y', 'z', 'cos_angle', 'v', 'vx', 'vy', 'vz', 'v_cps', 'intensity', 'id', 'label']
# columns = ['x', 'y', 'z', 'v', 'vx', 'vy', 'v_cps', 'intensity']
# df = pd.DataFrame(data, columns=columns)
# # 自适应文件名
# df.to_csv("data.csv", index=False)



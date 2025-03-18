import numpy as np
import os
import open3d as o3d

root_path = r"F:\Software\carla_tools\dataset\record_2024_1229_2143\vehicle.tesla.model3.master\carla4d_kitti_zrt"

pcd_path = os.path.join(root_path, "training", "velodyne")
label_path = os.path.join(root_path, "training", "label_2")
calib_path = os.path.join(root_path, "training", "calib")


def get_lidar(pcd_file):
    pcd = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 12)
    return pcd

def merge_multi_frame(pcd_path, label_path, calib_path, start_frame, frame_num=10):
    pcd_files = sorted(os.listdir(pcd_path))
    label_files = sorted(os.listdir(label_path))
    calib_files = sorted(os.listdir(calib_path))
    for i in range(start_frame, len(pcd_files) - frame_num):
        pcds = []
        labels = []
        calibs = []
        # 读取过去帧
        for j in range(0, -frame_num, -1):
            print(pcd_files[i + j])
            pcd = get_lidar(os.path.join(pcd_path, pcd_files[i + j]))
            pcds.append(pcd)
            # label = get_label(os.path.join(label_path, label_files[i + j]))
            # labels.append(label)
            # calib = get_calib(os.path.join(calib_path, calib_files[i + j]))
            # calibs.append(calib)
        pcds = np.concatenate(pcds, axis=0)
        
        break
    return pcds

if __name__ == "__main__":
    pcd = merge_multi_frame(pcd_path, label_path, calib_path, 44)[:, :3]
    print(pcd)
    # 可视化
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([o3d_pcd])
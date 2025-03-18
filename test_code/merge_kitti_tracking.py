import numpy as np
import os

dataset_root = r"M:\datasetsss\kitti_tracking\training"
velodyne_clip_path = os.path.join(dataset_root, "velodyne_clip")
doppler_velocity_path = os.path.join(dataset_root, "doppler_velocity")

save_path = os.path.join(dataset_root, "velodyne_clip_with_doppler_velocity")

for scene in range(21):
    scene = "{:04d}".format(scene)
    for frame in os.listdir(os.path.join(velodyne_clip_path, scene)):
        velodyne_clip = np.fromfile(os.path.join(velodyne_clip_path, scene, frame), dtype=np.float32).reshape(-1, 4)
        doppler_velocity = np.fromfile(os.path.join(doppler_velocity_path, scene, frame), dtype=np.float32).reshape(-1, 1)
        velodyne_clip_with_doppler_velocity = np.concatenate((velodyne_clip, doppler_velocity), axis=-1)
        if not os.path.exists(os.path.join(save_path, scene)):
            os.makedirs(os.path.join(save_path, scene))
        velodyne_clip_with_doppler_velocity.tofile(os.path.join(save_path, scene, frame))
        print(f"save {scene}/{frame}")
        
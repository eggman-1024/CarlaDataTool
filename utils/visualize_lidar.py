#!/usr/bin/python3

import argparse
import glob
import os.path
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
import open3d as o3d
from matplotlib import cm
import pandas as pd

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import ROOT_PATH

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
VID_RANGE2 = np.linspace(-5, 5, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (22, 196, 26),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


class PointcloudType(Enum):
        LIDAR = 0
        SEMANTIC_LIDAR = 1
        RADAR = 2
        LENGTH =3


class LidarVisualizer:
    def __init__(self, pointcloud_type: PointcloudType, source: str, frame: bool):
        self.pointcloud_type = pointcloud_type
        self.source = source
        self.pcd = o3d.geometry.PointCloud()
        self.bounds = []
        self.frame = frame

    def visualize(self):
        if self.source.endswith('.npy'):
            raw_pcd = np.load(self.source)
            print(raw_pcd)
            self.numpy_to_o3d(raw_pcd)
            #o3d.visualization.draw_geometries([self.pcd])
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(self.pcd)
            vis.get_render_option().background_color = [0,0,0]
            quaternion = np.array([0, 0, 0, 3.14])
            R = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
                
            view_control = vis.get_view_control()
            front_vector = R[:, 0].tolist()
            up_vector = R[:, 2].tolist()
            # set up/lookat/front vector to vis
            view_control.set_front(front_vector)
            view_control.set_up(up_vector)
            view_control.set_lookat(front_vector)
            # because I want get first person view, so set zoom value with 0.001, if set 0, there will be nothing on screen.
            view_control.set_zoom(0.01)
            vis.run()
            vis.destroy_window()


        else:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Carla Lidar')
            # vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 3
            vis.get_render_option().show_coordinate_frame = False
            vis.get_render_option().background_color = [0,0,0]
            self.add_open3d_axis(vis)
            files = sorted(glob.glob("{}/*.npy".format(self.source)))
            frame = 0
            for file in files:
                text_3d(file,(0,0,0),font_size=50)
                # vis.clear_geometries()
                for i in self.bounds:
                    vis.remove_geometry(i,False)
                    #vis.clear_3d_labels()
                self.bounds = []
                raw_pcd = np.load(file)
                self.numpy_to_o3d(raw_pcd)
                if frame == 0:
                    vis.add_geometry(self.pcd)
                    quaternion = np.array([0, 0.1, 0, 3])
                
                    R = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
                
                    view_control = vis.get_view_control()
                    front_vector = R[:, 0].tolist()
                    up_vector = R[:, 2].tolist()
                    # set up/lookat/front vector to vis
                    view_control.set_front(front_vector)
                    view_control.set_up(up_vector)
                    view_control.set_lookat(front_vector)
                    # because I want get first person view, so set zoom value with 0.001, if set 0, there will be nothing on screen.
                    view_control.set_zoom(0.01)

                for i in self.bounds:
                    vis.add_geometry(i,False)

                vis.update_geometry(self.pcd)
                vis.poll_events()
                vis.update_renderer()
                frame += 1
            vis.destroy_window()

    def numpy_to_o3d(self, numpy_cloud):
        if self.pointcloud_type == PointcloudType.LIDAR:
            # Isolate the intensity and compute a color for it
            intensity = np.zeros(numpy_cloud.shape[0])
            for i in range(numpy_cloud.shape[0]):
                intensity[i] = numpy_cloud[i].tolist()[-3]
            # intensity = numpy_cloud[:, -3]
            intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
            int_color = np.c_[
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

            # Isolate the 3D data
            points = np.zeros((numpy_cloud.shape[0], 3))
            for i in range(numpy_cloud.shape[0]):
                points[i] = numpy_cloud[i].tolist()[0:3]
            # points = numpy_cloud[:, 0:3]
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(int_color)
            return True
        elif self.pointcloud_type == PointcloudType.SEMANTIC_LIDAR:
            df = pd.DataFrame(numpy_cloud)
            def get_target_point_and_bound(df,index):
                actor_points = df[df.ObjIdx == index].values
                actor_tag = actor_points[0][5]
                actor_speed = df[df.ObjIdx == index].v.mean()
                actor_points = np.asarray(actor_points[:,0:3])
                actor_pcd = o3d.geometry.PointCloud()
                actor_pcd.points =o3d.utility.Vector3dVector(actor_points)
                if len(actor_points) < 4:
                    return None
                aabb = actor_pcd.get_axis_aligned_bounding_box()#.get_oriented_bounding_box()
                if actor_tag == 10:
                    aabb.color = (1,0,0)
                elif actor_tag == 4:
                    aabb.color = (0,1,0)
                else:
                    return None
                return [aabb,text_3d("{:.2f}".format(actor_speed),aabb.get_center(),(2,0,0),degree=10,font_size=20)]
            target_ids = df[(df.ObjTag == 4) | (df.ObjTag == 10)].ObjIdx.unique()
            for i in target_ids:
                res = get_target_point_and_bound(df,i)
                if res is None:
                    continue
                self.bounds.extend(res)
            # Read points
            points = np.array([numpy_cloud['x'], numpy_cloud['y'], numpy_cloud['z']]).T

            # Colorize the pointcloud based on the CityScapes color palette
            #labels = np.array(numpy_cloud['ObjTag'])
            #int_color = LABEL_COLORS[labels]
            intensity = numpy_cloud['v']
            #intensity = self.normalization(intensity)
            #intensity = abs(intensity)
            #intensity_col =  1 - np.log(intensity*10) / np.log(np.exp(-0.004 * 50))
            intensity_col = intensity
        
            cmap = cm.get_cmap('tab20c')
            piyg = cmap(np.arange(0,cmap.N)) 
            piyg_range = np.linspace(-15, 10, piyg.shape[0])
            int_color = np.c_[
                np.interp(intensity_col, piyg_range, piyg[:, 0]),
                np.interp(intensity_col, piyg_range, piyg[:, 1]),
                np.interp(intensity_col, piyg_range, piyg[:, 2])]
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(int_color)
            return True
        elif self.pointcloud_type == PointcloudType.RADAR:
            self.pcd.points = o3d.utility.Vector3dVector(numpy_cloud[:, 0:3])
            return True
        elif self.pointcloud_type == PointcloudType.LENGTH:
            # # 计算距离向量
            # length = np.linalg.norm(numpy_cloud[:, 0:3], axis=1)
            # # 拼接到原向量
            # new_points = np.c_[numpy_cloud[:, 0:3], length]
            # # 计算points到原点距离
            # #numpy_cloud[4] = np.sqrt(numpy_cloud[0] ** 2 + numpy_cloud[1] ** 2 + numpy_cloud[2] ** 2)
            # length = numpy_cloud[:, -1]
            # length_col = 1.0 - np.log(length) / np.log(np.exp(-0.004 * 100))
            # int_color = np.c_[
            #     np.interp(length_col, VID_RANGE, VIRIDIS[:, 0]),
            #     np.interp(length_col, VID_RANGE, VIRIDIS[:, 1]),
            #     np.interp(length_col, VID_RANGE, VIRIDIS[:, 2])]

            # # Isolate the 3D data
            # points = numpy_cloud[:, 0:3]
            # self.pcd.points = o3d.utility.Vector3dVector(points)
            # self.pcd.colors = o3d.utility.Vector3dVector(int_color)
            df = pd.DataFrame(numpy_cloud)
            df["length"] = np.linalg.norm(df[["x", "y", "z"]], axis=1)
            points = np.array([numpy_cloud['x'], numpy_cloud['y'], numpy_cloud['z']]).T
            length = df["length"]
            cmap = cm.get_cmap('hsv')
            piyg = cmap(np.arange(0,cmap.N)) 
            piyg_range = np.linspace(0, 80, piyg.shape[0])
            int_color = np.c_[
                np.interp(length, piyg_range, piyg[:, 0]),
                np.interp(length, piyg_range, piyg[:, 1]),
                np.interp(length, piyg_range, piyg[:, 2])]
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(int_color)


            
        else:
            return False
    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    def add_open3d_axis(self, vis):
        """Add a small 3D axis on Open3D Visualizer"""
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        axis.lines = o3d.utility.Vector2iVector(np.array([
            [0, 1],
            [0, 2],
            [0, 3]]))
        axis.colors = o3d.utility.Vector3dVector(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        vis.add_geometry(axis)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--type',
        default='lidar',
        help='Type of point cloud (lidar / semantic_lidar / radar/ length)'
    )
    argparser.add_argument(
        '--source',
        default=r'raw_data\record_2024_0929_1727\vehicle.tesla.model3.master\velodyne_semantic\0000002484.npy',
        type=str,
        help='File or folder source to visualize'
    )
    argparser.add_argument(
        "--frame",
        type=bool,
        default=False,
    )

    args = argparser.parse_args()
    pointcloud_type = PointcloudType.LIDAR
    if args.type == 'lidar':
        pointcloud_type = PointcloudType.LIDAR
    elif args.type == 'semantic_lidar':
        pointcloud_type = PointcloudType.SEMANTIC_LIDAR
    elif args.type == 'radar':
        pointcloud_type = PointcloudType.RADAR
    elif args.type == 'length':
        pointcloud_type = PointcloudType.LENGTH
    else:
        print("Not valid point cloud type")
        raise RuntimeError

    source = args.source
    if not os.path.exists(source):
        source = "{}/{}".format(ROOT_PATH, source)
        if not os.path.exists(source):
            print("File or folder not exist: {}".format(source))
            raise RuntimeError
    print("Read data from: {}".format(source))

    lidar_visualizer = LidarVisualizer(pointcloud_type, source, args.frame)
    lidar_visualizer.visualize()


def text_3d(text, pos, direction=None, degree=0.0, font='arial.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 255, 0))
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


if __name__ == "__main__":
    # execute only if run as a script
    main()


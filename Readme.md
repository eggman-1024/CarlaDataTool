<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">CarlaTool</h3>

<div align="center">

<!-- [![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE) -->

</div>

---

<p align="center"> Collect pointcloud data from CARLA and transform it to KITTI format.
    <br> 
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [🧐 About ](#-about-)
- [📝 File Description](#-file-description)
  - [Details](#details)
- [🏁 Getting Started ](#-getting-started-)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
- [🎈 Usage ](#-usage-)
  - [收集原始数据](#收集原始数据)
  - [数据集转换](#数据集转换)

## 🧐 About <a name = "about"></a>

本项目主要用于：
- 从[CARLA Simulator](https://carla.org/)(windows 0.9.13版本)收集仿真数据集，包括LiDAR点云及Camera图片。
- 将采集的raw_data转换为KITTI格式。

## 📝 File Description
```
├─config/                                     # 配置文件
│  ├─actor_settings_with_map                  
│  └─kitti_object
├─dataset/                                    # 数据集成品
│  └─record
│      └─vehicle.tesla.model3.master
│          └─carla360
│              ├─ImageSets
│              └─training
│                  ├─calib
│                  ├─image_2
│                  ├─label_2
│                  ├─status
│                  └─velodyne
├─label_tools/                                # 标签处理工具
│  ├─kitti_object
│  │  └─__pycache__
│  └─yolov5
│      └─__pycache__
├─raw_data/                                   # 原始录制数据
│  ├─record_2025_0315_2001
│  │  ├─others.world_0
│  │  ├─vehicle.tesla.model3.master
│  │  │  ├─image_2
│  │  │  └─velodyne_semantic
│  │  └─world
│  ├─......
│  └─......
├─recorder/                                   # carla对象
│  ├─agents
│  │  ├─navigation
│  │  └─tools
├─test_code/                                  # 测试代码
└─utils/                                      # 工具函数
```
### Details
---
- **config/**
  - **actor_settings_with_map/**: 包含了每个地图的actor, other_vehicles, other_pedestrians设置
  - **kitti_object/**: 暂时没用到
  - `world_config_template.json`: 包含每个地图的world设置
  - `sensor_config_template.json`: camera和LiDAR参数设置
  - 其它文件没用到
- **dataset/**
  - **record/vehicle.tesla.model3.master/carla360/**
      - **ImageSets/**: train, val, trainval划分
      - **training/**:    像KITTI数据集一样的格式
        - **calib/**:     calib文件
        - **image_2/**:   图片
        - **label_2/**:   标签
        - **status/**:    状态文件(该文件为自定义文件, 用于记录每一帧的timestamp, LiDAR的位姿, 自车位姿)
        - **velodyne/**： 点云
- **label_tools/**
  - **kitti_object/**: 转换为KITTI格式的工具代码
  - **yolov5/**: 转换为yolov5格式(暂时没用到)
  - `kitti_objects_label.py`: 将carla格式转换为KITTI格式的主要代码文件(deprecated)
  - `select_fragment2kitti.py`: **从raw_data中提取片段并转换为KITTI格式(用这个来转换数据集格式)**
  - `yolo_label.py`: 将carla格式转换为yolov5格式的主要代码文件(暂时没用到)
- **raw_data/**
  - **record_xxxx_xxxx_xxxx/**: 每个子文件夹就是一段连续的录制数据
- **recorder/** (该文件夹下主要构造了carla的对象和carla官方的API)
- **test_code/** (开发时的测试代码)
- **utils/** (开发中的工具函数, 基本无需更改)
- `data_recorder.py`: **用于收集原始数据**



## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

- 成功编译 Carla 0.9.13 windows版 
- 足够的硬盘空间(存储数据)



### Installing

- clone到本地

```bash
git clone https://github.com/eggman-1024/CarlaTool.git
```

- 创建并进入conda环境
  
```bash
conda create -n carla_913 python=3.8
conda activate carla_913
```

- 根据`requirements.txt`安装依赖包

```bash
pip install -r requirements.txt
```


## 🎈 Usage <a name="usage"></a>

### 收集原始数据
- 打开Carla
- 根据需求配置config
- 运行`data_recorder.py`
```bash
python data_recorder.py
```

### 数据集转换
运行`label_tools/select_fragment2kitti.py`
```bash
python label_tools/select_fragment2kitti.py
```


<!-- ## 🚀 Deployment <a name = "deployment"></a>

Add additional notes about how to deploy this on a live system. -->

<!-- ## ⛏️ Built Using <a name = "built_using"></a>

- [MongoDB](https://www.mongodb.com/) - Database
- [Express](https://expressjs.com/) - Server Framework
- [VueJs](https://vuejs.org/) - Web Framework
- [NodeJs](https://nodejs.org/en/) - Server Environment -->

<!-- ## ✍️ Authors <a name = "authors"></a>

- [@kylelobo](https://github.com/kylelobo) - Idea & Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References -->

import carla
import numpy as np
import random

def get_walker_bbox_dimensions(world):
    """
    获取所有行人的边界框尺寸并计算平均值。
    
    :param world: Carla 仿真环境的 world 对象
    :return: 所有行人边界框尺寸的平均值
    """
    # 随机生成一些行人
    spawn_points = world.get_map().get_spawn_points()
    # 生成vehicle
    vehicle_bps = world.get_blueprint_library().filter("vehicle.*")
    for bp in vehicle_bps:
        print(bp)
    walker_bp = random.choice(world.get_blueprint_library().filter("vehicle.*"))
    # walker_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
    walkers = []
    for _ in range(50):
        spawn_point = random.choice(spawn_points)
        walker = world.spawn_actor(walker_bp, spawn_point)
        walkers.append(walker)
        bounding_box = walker.bounding_box

        # 计算尺寸大小 (宽, 长, 高) = (2 * 半宽度, 2 * 半长度, 2 * 半高度)
        walker_size = np.array([bounding_box.extent.x * 2, 
                                bounding_box.extent.y * 2, 
                                bounding_box.extent.z * 2])
        
        # dimensions.append(walker_size)
        print(f"Walker ID {walker.id} dimensions (WxLxH): {walker_size}")


    dimensions = []

    # 遍历每个行人，获取其边界框
    for walker in walkers:
        # 获取 bounding box（包含尺寸信息）
        bounding_box = walker.bounding_box

        # 计算尺寸大小 (宽, 长, 高) = (2 * 半宽度, 2 * 半长度, 2 * 半高度)
        walker_size = np.array([bounding_box.extent.x * 2, 
                                bounding_box.extent.y * 2, 
                                bounding_box.extent.z * 2])
        
        dimensions.append(walker_size)
        print(f"Walker ID {walker.id} dimensions (WxLxH): {walker_size}")

    # 计算所有 walker 的尺寸平均值
    average_dimensions = np.mean(dimensions, axis=0)

    return average_dimensions

def main():
    # 连接到 Carla 服务器
    client = carla.Client('localhost', 2000)  # 假设 Carla 在本地运行
    client.set_timeout(10.0)  # 设置超时时间

    # 获取 Carla 仿真世界
    world = client.get_world()

    # 获取并计算所有行人的边界框尺寸平均值
    avg_bbox_dimensions = get_walker_bbox_dimensions(world)

    if avg_bbox_dimensions is not None:
        print(f"Average walker bbox dimensions (WxLxH): {avg_bbox_dimensions}")

if __name__ == "__main__":
    main()

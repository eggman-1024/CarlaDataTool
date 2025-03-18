import carla
import random
import time

def get_blueprint(world):
    """
    Get blueprint of the vehicle
    :param world: carla.World
    :return: carla.ActorBlueprint
    """
    blueprints = [bp for bp in world.get_blueprint_library().filter('vehicle*')]
    count = len(blueprints)
    print('Found {} blueprints'.format(count))
    return blueprints
    

def main():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bps = get_blueprint(world)
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    specator = world.get_spectator()
    specator.set_transform(carla.Transform(spawn_point.location + carla.Location(z=5), carla.Rotation(pitch=-90)))
    for bp in bps:
        print(bp)
        vehicle = world.spawn_actor(bp, spawn_point)
        time.sleep(3)
        vehicle.destroy()


if __name__ == "__main__":
    main() 
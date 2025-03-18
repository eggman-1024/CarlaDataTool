#!/usr/bin/python3
import time
import carla
import sys
import random
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())
from utils.transform import carla_transform_to_transform


def show_spawn_points(world: carla.World, debug_helper: carla.DebugHelper):
    spawn_points = world.get_map().get_spawn_points()
    for i in range(len(spawn_points)):
        carla_location = spawn_points[i].location
        debug_helper.draw_string(carla.Location(carla_location.x,
                                                carla_location.y + 1.0,
                                                carla_location.z + 1.0),
                                 "SpawnPoint-{}".format(i))
        bbox = carla.BoundingBox(spawn_points[i].location, carla.Vector3D(0.1, 0.1, 0.1))
        debug_helper.draw_box(bbox,
                              rotation=carla.Rotation())
    for carla_actor in world.get_actors():
        if carla_actor.type_id == 'traffic.traffic_light':
            cid = carla_actor.id
            carla_location = carla_actor.get_location()
            carla_rotation = carla_actor.get_transform().rotation
            bbox = carla.BoundingBox(carla_location,
                                     carla.Vector3D(2,
                                                    2,
                                                    2))
            debug_helper.draw_string(carla.Location(carla_location.x,
                                                    carla_location.y + 1.0,
                                                    carla_location.z + 1.0),
                                     "TrafficLight-{}".format(cid),
                                     color=carla.Color(0, 255, 0))
            debug_helper.draw_box(bbox,
                                  rotation=carla_rotation,
                                  color=carla.Color(0, 255, 0))
            
def show_bbox(carla_actor: carla.Actor, debug_helper: carla.DebugHelper):
    carla_location = carla_actor.get_location()
    carla_rotation = carla_actor.get_transform().rotation
    # bbox = carla_actor.bounding_box
    bbox = carla.BoundingBox(carla_location,
                             carla.Vector3D(2,2,2))
    debug_helper.draw_box(bbox,
                          rotation=carla_rotation,
                          color=carla.Color(255, 0, 0),
                          thickness=0.1,
                          life_time=0)
    debug_helper.draw_string(carla.Location(carla_location.x,
                                            carla_location.y + 1.0,
                                            carla_location.z + 1.0),
                             "Actor-{}".format(carla_actor.id),
                             color=carla.Color(255, 0, 0))


def main():
    host = '127.0.0.1'
    port = 2000
    carla_client = carla.Client(host, port)
    carla_client.set_timeout(2.0)
    carla_world = carla_client.get_world()
    debug = carla_world.debug
    spector = carla_world.get_spectator()

    show_spawn_points(carla_world, debug)
    cyclist_bps = ["vehicle.harley-davidson.low_rider", 
                    "vehicle.yamaha.yzf", 
                    "vehicle.kawasaki.ninja", 
                    "vehicle.bh.crossbike", 
                    "vehicle.gazelle.omafiets", 
                    "vehicle.diamondback.century", 
                    "vehicle.vespa.zx125"]
    spawn_points = carla_world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    specator = carla_world.get_spectator()
    specator.set_transform(carla.Transform(spawn_point.location + carla.Location(z=5), carla.Rotation(pitch=-90)))
    vehicle = carla_world.spawn_actor(carla_world.get_blueprint_library().find(cyclist_bps[0]), spawn_point)
    show_bbox(vehicle, debug)
    time.sleep(3)
    vehicle.destroy()

    try:
        while True:
            carla_transform = spector.get_transform()
            print(carla_transform_to_transform(carla_transform))
            time.sleep(0.1)
    except KeyboardInterrupt:
        carla_client.reload_world()


if __name__ == "__main__":
    # execute only if run as a script
    main()

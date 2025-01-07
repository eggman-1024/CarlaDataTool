import argparse
import pickle
import json
import os
import signal
import time

import carla

from param import *
from recorder.actor_tree import ActorTree
from utils.transform import Transform, Location, Rotation
from utils.transform import transform_to_carla_transform,carla_vec3d_to_vec3d,carla_transform_to_transform

sig_interrupt = False


def signal_handler(signal, frame):
    global sig_interrupt
    sig_interrupt = True


class DataRecorder:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.carla_client = carla.Client(self.host, self.port)
        self.carla_client.set_timeout(10.0)
        self.world = self._get_world()
        self.tm = self.carla_client.get_trafficmanager()
        self.debug_helper = self.world.debug
        self.record_name = None
        self.base_save_dir = None
        self.world_config_file = "{}/config/{}".format(ROOT_PATH, args.world_config_file)
        self.actor_tree = ActorTree(self.world)
        self.frame_total = -1
        self.frame_step = 1

    def _get_world(self) -> carla.World:
        return self.carla_client.get_world()

    def destroy(self):
        self.actor_tree.destroy()

    def set_traffic_light_time(self, traffic_light_setting):
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if isinstance(actor, carla.TrafficLight):
                actor.set_red_time(traffic_light_setting["red_time"])
                actor.set_green_time(traffic_light_setting["green_time"])
                actor.set_yellow_time(traffic_light_setting["yellow_time"])

    def setting_world_and_actors(self, json_file, index):
        with open(json_file) as handle:
            json_settings = json.loads(handle.read())["maps"][index]
            self.carla_client.load_world(json_settings["map"])
            settings = self.world.get_settings()
            settings.synchronous_mode = True

            if json_settings["spectator_pose"] is not None:
                pose = json_settings["spectator_pose"]
                spectator = self.world.get_spectator()
                spectator_transform = Transform(Location(pose["x"], pose["y"], pose["z"]),
                                                Rotation(roll=pose["roll"], pitch=pose["pitch"], yaw=pose["yaw"]))
                spectator.set_transform(transform_to_carla_transform(spectator_transform))

            # Make sure fixed_delta_seconds <= max_substep_delta_time * max_substeps
            world_settings = json_settings["world_settings"]
            settings.fixed_delta_seconds = world_settings["fixed_delta_seconds"]
            settings.substepping = True
            settings.max_substep_delta_time = world_settings["max_substep_delta_time"]
            settings.max_substeps = world_settings["max_substeps"]
            self.world.apply_settings(settings)
            self.tm.set_synchronous_mode(True)

            self.frame_total = json_settings["frame_total"]
            self.frame_step = json_settings["frame_step"]

            self.set_traffic_light_time(json_settings["traffic_light_setting"])

            actor_config_file = json_settings["actor_settings"]
            self.record_name = time.strftime("%Y_%m%d_%H%M", time.localtime())
            self.base_save_dir = "{}/record_{}".format(RAW_DATA_PATH, self.record_name)
            self.actor_tree = ActorTree(self.world,
                                        "{}/config/{}".format(ROOT_PATH,
                                                            actor_config_file),
                                        self.base_save_dir)
            self.actor_tree.init()
    def record_vehicles_speed(self,frame_id):
        self.speeds[frame_id]={}
        for i in self.obj:
            trans = carla_transform_to_transform(i.get_transform())
            trans.location = Location(0, 0, 0)
            vel_world = carla_vec3d_to_vec3d(i.get_velocity())
            vel_vehicle = trans.inv_transform(vel_world)
            trans = carla_transform_to_transform(i.get_transform())
            self.speeds[frame_id][i.id] = [vel_vehicle,trans]
    def save_vehicles_spped(self):
        with open("./vehicles.bin",'wb') as f:
            pickle.dump(self.speeds,f)

    def start_record(self, index):
        self.setting_world_and_actors(self.world_config_file, index)
        os.makedirs(self.base_save_dir, exist_ok=True)
        print("yes2")
        carla_logfile = "{}/carla_raw_record.log".format(self.base_save_dir)
        self.carla_client.start_recorder(carla_logfile)
        '''
        for i in self.actor_tree.node_list:
            try:
                print(i.get_actor().get_carla_actor())
            except:
                pass
        self.obj = []
        self.speeds = {}
        world = self.carla_client.get_world()
        for i in world.get_actors():
            try:
                v = i.get_velocity()
                self.obj.append(i)
            except:
                pass
        for i in self.actor_tree.node_list:
            actor = i.get_actor().get_carla_actor()
            if actor != None:
                try:
                    v = actor.get_velocity()
                    print(actor)
                    self.obj.append(actor)
                except:
                    pass
        '''
        print("yes3")
        try:
            total_frame_count = 0
            while True:
                print("----------")
                # Tick Control
                self.actor_tree.tick_controller()
                # Tick World
                tick_s = time.time()
                frame_id = self.world.tick(seconds=60.0)
                world_snapshot = self.world.get_snapshot()
                timestamp = world_snapshot.timestamp.elapsed_seconds
                print("World Tick -> FrameID: {} Timestamp: {} Cost: {:.3f}s".format(frame_id,
                                                                                     timestamp,
                                                                                     time.time()-tick_s))
                # Save data to disk
                if total_frame_count % self.frame_step == 0:
                    save_s = time.time()
                    self.actor_tree.tick_data_saving(frame_id, timestamp)
                    #self.record_vehicles_speed(frame_id)
                    print("Raw data saved, cost {:.3f}s".format(time.time()-save_s))

                global sig_interrupt
                if sig_interrupt:
                    #self.save_vehicles_spped()
                    print("Exit step, wait 2 seconds...")
                    time.sleep(2.0)
                    break

                total_frame_count += 1
                if total_frame_count >= self.frame_total:
                    #self.save_vehicles_spped()
                    time.sleep(2.0)
                    break

        except KeyboardInterrupt:
            #self.save_vehicles_spped()
            print("User interrupt, exit...")
        except Exception as e:
            print("Error: {}".format(e))
        else:
            #self.save_vehicles_spped()
            print("Unhandled error: reload the world and exit...")
        self.destroy()
        self.carla_client.reload_world()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-w', '--world_config_file',
        metavar='W',
        default='world_config_template.json',
        type=str,
        help='World configuration file')
    
    args = argparser.parse_args()
    with open("{}/config/{}".format(ROOT_PATH, args.world_config_file)) as handle:
        num_maps = len(json.loads(handle.read())["maps"])
    for i in range(num_maps):
        print("Start recording map {}".format(i))
        data_recorder = DataRecorder(args)
        # print("yes")
        data_recorder.start_record(i)

def gen_man(num):
    import random
    client = carla.Client("127.0.0.1", 2000)
    spawn_points = []
    world = client.get_world()
    blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
    for i in range(num):
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()
        if (spawn_point.location != None):
            spawn_points.append(spawn_point)
    for spawn_point in spawn_points:
        bp = random.choice(blueprints)
        carla_actor = world.spawn_actor(bp, spawn_point)
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        ai_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), carla_actor)
        ai_controller.start()
        ai_controller.go_to_location(world.get_random_location_from_navigation())
        ai_controller.set_max_speed(1 + random.random())

if __name__ == "__main__":
    # execute only if run as a script
    main()

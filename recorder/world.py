#!/usr/bin/python3
import os
import pickle
import carla
from dataclasses import dataclass

from recorder.actor import PseudoActor
from utils.label_types import *
from utils.transform import carla_bbox_to_bbox, carla_transform_to_transform, carla_vec3d_to_vec3d
from utils.geometry_types import Location, Vector3d, BoundingBox


class WorldActor(PseudoActor):
    def __init__(self, uid, carla_world: carla.World, base_save_dir: str):
        super().__init__(uid, self.get_type_id(), None)
        self.save_dir = "{}/{}_{}".format(base_save_dir, self.get_type_id(), uid)
        self.carla_world = carla_world

    def save_to_disk(self, frame_id, timestamp, debug=False):
        # TODO: Save all object bbox in world
        # Frame Timestamp CityObjectLabel carla_id location rotation box_location box_extent
        object_labels = []

        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Vehicles)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Pedestrians)

        carla_actors = self.carla_world.get_actors()
        for carla_actor in carla_actors:
            if carla_actor.type_id.startswith('vehicle') \
                    or carla_actor.type_id.startswith('walker'):
                # fix bbox width
                if carla_actor.bounding_box.extent.y == 0:
                    bbox = carla_bbox_to_bbox(BoundingBox(carla_actor.bounding_box.location, 
                                                          Vector3d(carla_actor.bounding_box.extent.x, 
                                                                   self.fix_bbox_width(carla_actor), 
                                                                   carla_actor.bounding_box.extent.z)))
                else:
                    bbox = carla_bbox_to_bbox(carla_actor.bounding_box)
                transform = carla_transform_to_transform(carla_actor.get_transform())
                if carla_actor.type_id.startswith('walker'):
                    label_type = 'pedestrian'
                elif carla_actor.type_id.startswith('vehicle') and carla_actor.semantic_tags == [10,4]:
                    label_type = 'cyclist'
                else:
                    label_type = 'vehicle'
                    
                object_labels.append(ObjectLabel(frame=frame_id,
                                                 timestamp=timestamp,
                                                 label_type=label_type,
                                                 carla_id=carla_actor.id,
                                                 transform=transform,
                                                 bounding_box=bbox))

        if len(object_labels) == 0:
            return False

        os.makedirs(self.save_dir, exist_ok=True)
        with open('{}/{:0>10d}.pkl'.format(self.save_dir, frame_id), 'wb') as pkl_file:
            pickle.dump(obj=object_labels, file=pkl_file)
        if debug:
            print("WorldObjectsLabel: Frame: {} Total counts: {}".format(frame_id, len(object_labels)))
        return True
    
    def fix_bbox_width(self, carla_actor):
        bounding_box = carla_actor.bounding_box
        l, w, h = bounding_box.extent.x, bounding_box.extent.y, bounding_box.extent.z
        if l == 0.8214218020439148:
            fixed_w = 0.18625812232494354
        elif l == 1.1778701543807983:
            fixed_w = 0.38183942437171936
        elif l == 1.1047229766845703:
            fixed_w = 0.43351709842681885
        elif l == 0.9023334980010986:
            fixed_w = 0.42784383893013
        elif l == 1.0166761875152588:
            fixed_w = 0.4012899398803711
        elif l == 0.9177202582359314:
            fixed_w = 0.42784383893013
        elif l == 0.7436444163322449:
            fixed_w = 0.42962872982025146

        return fixed_w

    def get_type_id(self):
        return 'others.world'

    def get_save_dir(self):
        return self.save_dir

    def get_carla_transform(self) -> carla.Transform:
        return carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))

    def get_env_objects_labels(self, frame, timestamp, object_type: carla.CityObjectLabel) -> list:
        object_labels = []
        if object_type == carla.CityObjectLabel.Vehicles:
            label_type = 'vehicle'
        elif object_type == carla.CityObjectLabel.Pedestrians:
            label_type = 'pedestrian'
        else:
            label_type = 'any'
        env_objects = self.carla_world.get_environment_objects(object_type=object_type)
        for env_object in env_objects:
            transform = carla_transform_to_transform(env_object.transform)
            bbox_extent = Vector3d(env_object.bounding_box.extent.x,
                                   env_object.bounding_box.extent.y,
                                   env_object.bounding_box.extent.z)
            object_labels.append(ObjectLabel(frame=frame,
                                             timestamp=timestamp,
                                             label_type=label_type,
                                             carla_id=env_object.id,
                                             transform=transform,
                                             bounding_box=BoundingBox(Location(0, 0, 0), bbox_extent)))
        return object_labels

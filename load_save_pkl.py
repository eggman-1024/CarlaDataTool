import pickle
import os

filepath = r"raw_data\record_2025_0108_1530\others.world_0"
filenames = os.listdir(filepath)

save_path = r"raw_data\record_2025_0108_1530\world"
os.makedirs(save_path, exist_ok=True)


for filename in filenames:
    print(f"filename:{filename}")
    file_path = os.path.join(filepath, filename)
    with open(file_path, 'rb') as f:
        data_list = pickle.load(f)

        data_new_list = []

        for data in data_list:
            # print(data)

            frame = data.frame
            timestamp = data.timestamp
            label_type = data.label_type
            id = data.carla_id
            x = data.transform.location.x
            y = data.transform.location.y
            z = data.transform.location.z
            pitch = data.transform.rotation.pitch
            yaw = data.transform.rotation.yaw
            roll = data.transform.rotation.roll
            box_x = data.bounding_box.location.x
            box_y = data.bounding_box.location.y
            box_z = data.bounding_box.location.z
            box_dx = data.bounding_box.extent.x
            box_dy = data.bounding_box.extent.y
            box_dz = data.bounding_box.extent.z
            box_pitch = data.bounding_box.rotation.pitch
            box_yaw = data.bounding_box.rotation.yaw
            box_roll = data.bounding_box.rotation.roll

            # print(f"frame:{frame}, timestamp:{timestamp}, label_type:{label_type}, id:{id}, x:{x}, y:{y}, z:{z}, pitch:{pitch}, yaw:{yaw}, roll:{roll}, box_x:{box_x}, box_y:{box_y}, box_z:{box_z}, box_dx:{box_dx}, box_dy:{box_dy}, box_dz:{box_dz}, box_pitch:{box_pitch}, box_yaw:{box_yaw}, box_roll:{box_roll}")
            # print(f"type_frame:{type(frame)}, type_timestamp:{type(timestamp)}, type_label_type:{type(label_type)}, type_id:{type(id)}, type_x:{type(x)}, type_y:{type(y)}, type_z:{type(z)}, type_pitch:{type(pitch)}, type_yaw:{type(yaw)}, type_roll:{type(roll)}, type_box_x:{type(box_x)}, type_box_y:{type(box_y)}, type_box_z:{type(box_z)}, type_box_dx:{type(box_dx)}, type_box_dy:{type(box_dy)}, type_box_dz:{type(box_dz)}, type_box_pitch:{type(box_pitch)}, type_box_yaw:{type(box_yaw)}, type_box_roll:{type(box_roll)}")
            
            data_new_list.append([frame, timestamp, label_type, id, x, y, z, pitch, yaw, roll, box_x, box_y, box_z, box_dx, box_dy, box_dz, box_pitch, box_yaw, box_roll])
        
        # 保存为csv
        save_file_path = os.path.join(save_path, filename.replace(".pkl", ".csv"))
        with open(save_file_path, 'w') as f:
            for data_new in data_new_list:
                data_str = ",".join([str(data) for data in data_new])
                f.write(data_str + "\n")

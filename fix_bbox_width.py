import numpy as np
import pickle
import os
from tqdm import tqdm
from param import ROOT_PATH

def read_world_pkl(pkl_file_path):
    """
    读取并加载 .pkl 文件的数据
    :param pkl_file_path: str, .pkl 文件的路径
    :return: 数据对象
    """
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
        return data

def write_world_pkl(pkl_file_path, data):
    """
    将数据保存为 .pkl 文件
    :param pkl_file_path: str, .pkl 文件的路径
    :param data: 数据对象
    :return: None
    """
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(data, f)
    
def main():
    dataset_dir_list = [r"raw_data\record_2025_0106_2019", r"raw_data\record_2025_0106_2025", r"raw_data\record_2025_0106_2032", r"raw_data\record_2025_0106_2038", r"raw_data\record_2025_0106_2045", r"raw_data\record_2025_0106_2051"]
    for dataset_dir in dataset_dir_list:
        world_pkl_path = os.path.join(dataset_dir, "others.world_0")
        save_world_pkl_path = os.path.join(dataset_dir, "others.world_1")
        filenames = sorted(os.listdir(world_pkl_path))
        for filename in tqdm(filenames, desc=f"{dataset_dir}: Fixing bounding box width", colour="green"):
            pkl_file_path = os.path.join(world_pkl_path, filename)
            save_pkl_file_path = os.path.join(save_world_pkl_path, filename)
            if not os.path.exists(save_world_pkl_path):
                os.makedirs(save_world_pkl_path)
            datas = read_world_pkl(pkl_file_path)
            if datas is not None:
                for data in datas:
                    if data.label_type != "cyclist":
                        continue
                    bounding_box = data.bounding_box
                    l, w, h = bounding_box.extent.x, bounding_box.extent.y, bounding_box.extent.z
                    if w == 0:
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

                        data.bounding_box.extent.y = fixed_w
                    write_world_pkl(save_pkl_file_path, datas)

if __name__ == "__main__":
    main()
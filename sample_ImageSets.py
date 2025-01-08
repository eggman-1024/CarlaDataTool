import os
import random

def sample_dataset(origin_path, sample_path, train_val_flag=False):
    origin_train_file = os.path.join(origin_path, "train.txt")
    origin_val_file = os.path.join(origin_path, "val.txt")

    sample_train_file = os.path.join(sample_path, "train.txt")
    sample_val_file = os.path.join(sample_path, "val.txt")

    sample_train_lines = []
    sample_val_lines = []
    with open(origin_train_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        # 随机采样一半数据
        sample_train_lines = random.sample(lines, len(lines) // 2)
    with open(origin_val_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        # 随机采样一半数据
        sample_val_lines = random.sample(lines, len(lines) // 2)

    with open(sample_train_file, 'w') as f:
        for line in sample_train_lines:
            f.write(f"{line}\n")
    with open(sample_val_file, 'w') as f:
        for line in sample_val_lines:
            f.write(f"{line}\n")

    if train_val_flag:
        sample_trainval_file = os.path.join(sample_path, "trainval.txt")
        sample_trainval_lines = sample_train_lines + sample_val_lines
        with open(sample_trainval_file, 'w') as f:
            for line in sample_trainval_lines:
                f.write(f"{line}\n")



if __name__ == "__main__":
    origin_path = r"F:\Software\carla_tools\dataset\record\ImageSets"
    sample_path = r"F:\Software\carla_tools\dataset\record\sample_ImageSets"

    os.mkdir(sample_path)
    sample_dataset(origin_path, sample_path, train_val_flag=True)


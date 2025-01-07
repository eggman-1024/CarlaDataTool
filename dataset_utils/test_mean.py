import numpy as np
import pandas as pd
import pickle
import os



def generate_training_data(input_csv, output_folder, max_value=1e+10, min_value=-1e+10):
    # 加载原始数据前10行
    df = pd.read_csv(input_csv)[:10]

    # 初始化存储结果的列表
    training_data = []
    csv_data = []  # 用于保存 CSV 格式数据

    # 逐行处理数据
    for i in range(len(df) - 1):
        # 输入特征
        row = df.iloc[i]
        x = row["x"]
        y = row["y"]
        z = row["z"]
        vx = row["vx"]
        vy = row["vy"]
        v_cps = row["v_cps"]
        intensity = row["intensity"]
        
        # 将特征和标签保存为一条训练数据
        inputs = [x, y, z, vx, vy, v_cps]
        outputs = [intensity]

        # 数据清洗：移除输入或输出中任意值超出范围的训练数据
        if any(x <= min_value or x >= max_value for x in inputs + outputs):
            continue

        training_data.append({
            "inputs": inputs,
            "outputs": outputs
        })

        # 用于保存 CSV 格式的训练数据
        csv_data.append(inputs + outputs)

    # 保存为 .pkl 文件
    try:
        output_pkl = os.path.join(output_folder, "training_data.pkl")
        with open(output_pkl, 'wb') as file:
            pickle.dump(training_data, file)
        print(f"训练数据已成功保存为 {output_pkl}")
    except Exception as e:
        print(f"保存失败: {e}")

    # 保存为 .csv 文件
    try:
        csv_df = pd.DataFrame(csv_data, columns=["v_x", "v_y", "acc_x", "acc_y", "delta_x", "delta_y", "delta_yaw", "curr_throttle", "curr_steer", "throttle", "steer"])
        csv_file = output_pkl.replace(".pkl", ".csv")  # 根据 pkl 文件路径生成 csv 文件路径
        csv_df.to_csv(csv_file, index=False)
        print(f"训练数据已成功保存为 {csv_file}")
    except Exception as e:
        print(f"保存失败: {e}")


generate_training_data("data.csv", ".")

with open("./training_data.pkl", 'rb') as f:
    data = pickle.load(f)
all_inputs = np.array([sample["inputs"] for sample in data])
print(f"all_inputs:{all_inputs}")

mean = all_inputs.mean(axis=0)
print(f"mean:{mean}")
std = all_inputs.std(axis=0) + 1e-8  # 避免除以零
print(f"std:{std}")
for sample in data:
    sample["inputs"] = (sample["inputs"] - mean) / std
    print(f"sample:{sample}")
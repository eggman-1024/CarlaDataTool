import pickle

def load_pkl_file(pkl_file_path):
    """
    读取并加载 .pkl 文件的数据
    :param pkl_file_path: str, .pkl 文件的路径
    :return: 数据对象
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            print(f"pkl_file_path:{pkl_file_path}") 
            data = pickle.load(f)
            print(f"data:{type(data)}")
            return data
    except Exception as e:
        return None

def main():
    # 修改为你想要读取的 .pkl 文件的路径
    pkl_file_path = r'F:\Software\carla_tools\raw_data\record_2025_0106_2038\others.world_1\0000001085.pkl'
    # pkl_file_path = r'F:\Software\carla_tools\raw_data\record_2024_1229_2143\others.world_0\0000000062.pkl'
    # pkl_file_path = r"M:\datasetsss\carla_4d\record_2024_0928_1819\others.world_0\0000000160.pkl"

    # 读取并打印 .pkl 文件中的内容
    datas = load_pkl_file(pkl_file_path)
    if datas is not None:
        # print(f"datas:{datas[:10]}")
        print(f"len:{len(datas)}")
        # 统计每个data的label_type，看看一共有多少种，每种是什么
        label_types = {}
        for data in datas:
            print(f"data:{data}")
            label_type = data.label_type
            if label_type not in label_types:
                label_types[label_type] = 1
            else:
                label_types[label_type] += 1
        print("label_types: ", label_types)
        # 总数
        print("total: ", sum(label_types.values()))
    print(f"pkl_file_path:{pkl_file_path}")
if __name__ == "__main__":
    main()

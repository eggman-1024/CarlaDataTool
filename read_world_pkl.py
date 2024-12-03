import pickle

def load_pkl_file(pkl_file_path):
    """
    读取并加载 .pkl 文件的数据
    :param pkl_file_path: str, .pkl 文件的路径
    :return: 数据对象
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
            print("文件加载成功！")
            return data
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def main():
    # 修改为你想要读取的 .pkl 文件的路径
    pkl_file_path = r'raw_data\record_2024_0929_1727\others.world_0\0000002384.pkl'

    # 读取并打印 .pkl 文件中的内容
    datas = load_pkl_file(pkl_file_path)
    if datas is not None:
        print("文件内容:")
        for data in datas:
            print(data)

if __name__ == "__main__":
    main()

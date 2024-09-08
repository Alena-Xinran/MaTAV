import pickle

# 文件路径
file_path = '/home/xrl/MultiEMO-ACL2023-main/Data/IEMOCAP/TextFeatures.pkl'

# 以二进制读取模式打开文件
with open(file_path, 'rb') as file:
    # 加载 pickle 数据
    data = pickle.load(file)

# 检查数据是否为字典类型
if isinstance(data, dict):
    # 遍历字典并打印键和值
    for key, value in data.items():
        print(f"Key: {key}, Value: {value}")
else:
    print("Loaded data is not a dictionary.")

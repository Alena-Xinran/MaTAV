import pickle

# �ļ�·��
file_path = '/home/xrl/MultiEMO-ACL2023-main/Data/IEMOCAP/TextFeatures.pkl'

# �Զ����ƶ�ȡģʽ���ļ�
with open(file_path, 'rb') as file:
    # ���� pickle ����
    data = pickle.load(file)

# ��������Ƿ�Ϊ�ֵ�����
if isinstance(data, dict):
    # �����ֵ䲢��ӡ����ֵ
    for key, value in data.items():
        print(f"Key: {key}, Value: {value}")
else:
    print("Loaded data is not a dictionary.")

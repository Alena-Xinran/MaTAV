from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pickle
from pathlib import Path
import pandas as pd
'''
label index mapping = {'happiness': 0, 'sadness': 1, 'neutral': 2, 'anger': 3, 'excitement': 4, 'frustration': 5}
'''
class IEMOCAPDataset(Dataset):

    def __init__(self, train = True):
        _, self.videoSpeakers, self.videoLabels, _, _, _, _, self.trainVid,\
        self.testVid = pickle.load(open('/home/xrl/MultiEMO-ACL2023-main/Data/IEMOCAP/Speakers.pkl', 'rb'), encoding='latin1')

        '''
        Textual features are extracted using pre-trained EmoBERTa. If you want to extract textual
        features on your own, please visit https://github.com/tae898/erc
        '''
        self.videoText = pickle.load(open('/home/xrl/MultiEMO-ACL2023-main/Data/IEMOCAP/TextFeatures.pkl', 'rb'))
        self.videoAudio = pickle.load(open('/home/xrl/MultiEMO-ACL2023-main/Data/IEMOCAP/AudioFeatures.pkl', 'rb'))
        self.videoVisual = pickle.load(open('/home/xrl/MultiEMO-ACL2023-main/Data/IEMOCAP/VisualFeatures.pkl', 'rb'))

        self.trainVid = sorted(self.trainVid)
        self.testVid = sorted(self.testVid)

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)


    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(np.array(self.videoText[vid])),\
            torch.FloatTensor(np.array(self.videoAudio[vid])),\
                torch.FloatTensor(np.array(self.videoVisual[vid])),\
                    torch.FloatTensor(np.array([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]])),\
                        torch.FloatTensor(np.array([1]*len(self.videoLabels[vid]))),\
                            torch.LongTensor(np.array(self.videoLabels[vid]))


    def __len__(self):
        return self.len


    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        max_len=256
        output = []
        for i in dat:
            temp = dat[i].values
            if i <= 3:  # 对于文本、音频和视觉特征
                padded_sequence = pad_sequence([temp[j] for j in range(len(temp))], padding_value=0)
                # print(f"Feature {i} padded sequence shape: {padded_sequence.shape}")  # 打印维度
                # sequences = [temp[j] for j in range(len(temp))]
                # sequences = [seq[:max_len] if len(seq) > max_len else torch.cat((seq, torch.zeros(max_len - len(seq), seq.size(1))), dim=0) if len(seq) < max_len else seq for seq in sequences]
                # padded_sequence = pad_sequence(sequences, batch_first=True, padding_value=0)
                # print(f"Feature {i} padded sequence shape: {padded_sequence.shape}")
                output.append(padded_sequence)
            elif i <= 4:  # 对于说话者信息
                padded_sequence = pad_sequence([temp[j] for j in range(len(temp))], True, padding_value=0)
                #print(f"Feature {i} (Speaker info) padded sequence shape: {padded_sequence.shape}")  # 打印维度
                output.append(padded_sequence)
            elif i <= 5:  # 对于标签
                padded_sequence = pad_sequence([temp[j] for j in range(len(temp))], True, padding_value=-1)
                #print(f"Feature {i} (Labels) padded sequence shape: {padded_sequence.shape}")  # 打印维度
                output.append(padded_sequence)
        save_path = '/home/xrl/MultiEMO-ACL2023-main/Dataset/data-iemocap.csv'
        dat.to_csv(save_path, index=False)
        return output

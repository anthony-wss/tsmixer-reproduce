import pandas as pd
from torch.utils.data import Dataset
import numpy as np

def standardization(arr):
    return (arr - np.mean(arr)) / np.std(arr)

class ETDataset(Dataset):
    def __init__(self, L=24*4, T=24*4, flag='train', name='ETTh1'):
        self.label_len = L
        self.pred_len = T
        assert flag in ['train', 'test', 'val']
        assert name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.name = name
        self.C = 7
        self.__read_data__()
    
    def __read_data__(self):
        df = pd.read_csv(f"tsmixer/data/ETDataset/ETT-small/{self.name}.csv")
        cols = df.columns[1:]
        data = df[cols].values

        if self.name[-2] == 'h':
            border_sta = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
            border_end = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            num_data = 12 * 30 * 24 + 8 * 30 * 24
        else:
            border_sta = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
            border_end = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            num_data = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
        
        # chunks = []
        # chunk_size = self.label_len + self.pred_len
        # for i in range(num_data - chunk_size + 1):
        #     chunks.append(data[i : i+chunk_size, :])

        # np.random.seed(197)
        # np.random.shuffle(chunks)

        # num_chunks = len(chunks)
        # border_sta = [0, int(num_chunks*0.6), int(num_chunks*0.8)]
        # border_end = [int(num_chunks*0.6), int(num_chunks*0.8), int(num_chunks*1)]

        border_sta = border_sta[self.set_type]
        border_end = border_end[self.set_type]
        self.data = data[border_sta:border_end]

        for i in range(self.data.shape[1]):
            self.data[:, i] = standardization(self.data[:, i])
        
        # self.data = chunks[border_sta:border_end]
    
    def __getitem__(self, index):
        # return self.data[index][:self.label_len, :], self.data[index][self.label_len:, :]
        return self.data[index:index + self.label_len], self.data[index + self.label_len:index + self.label_len + self.pred_len]

    def __len__(self):
        # return len(self.data)
        return len(self.data) - self.label_len - self.pred_len + 1
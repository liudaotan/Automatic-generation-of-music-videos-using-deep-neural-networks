import torch as tr
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MusicDataLoader(Dataset):

    def __init__(self, root):
        self.datalist = np.load(root, allow_pickle=True)

    def __getitem__(self, idx):
        item = self.datalist[idx]
        data = tr.tensor(item[0])
        label = item[1]
        return data, tr.Tensor(label)

    def __len__(self):
        return len(self.datalist)


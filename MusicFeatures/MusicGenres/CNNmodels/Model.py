import torch as tr
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
import librosa
from prefetch_generator import BackgroundGenerator
import time

class CnnModel(tr.nn.Module):
    def __init__(self,num_class=8):
        super(CnnModel, self).__init__()
        #data (40,368)
        self.conv1 = tr.nn.Sequential(
            tr.nn.Conv2d(1, 64, 3, 1), #(38, 366)
            tr.nn.ReLU(True),
            tr.nn.Conv2d(64, 64, 3, 1), #(36, 364)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(2), #(18, 182)
            tr.nn.BatchNorm2d(64),
        )
        self.conv2 = tr.nn.Sequential(
            tr.nn.Conv2d(64, 128, 3, 1, 1), #(18, 182)
            tr.nn.ReLU(True),
            tr.nn.Conv2d(128, 128, 3, 1), #(16, 180)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(2), #(8, 90)
            tr.nn.BatchNorm2d(128),
        )
        self.conv3 = tr.nn.Sequential(
            tr.nn.Conv2d(128, 128, 3, 1), #(6, 88)
            tr.nn.ReLU(True),
            tr.nn.Conv2d(128, 128, 3, 1), #(4, 86)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(2), #(2, 43)
            tr.nn.BatchNorm2d(128),
        )
        self.linear = tr.nn.Sequential(
            tr.nn.Linear(2*43*128,4096),
            tr.nn.ReLU(True),
            tr.nn.Dropout(0.2),
            tr.nn.Linear(4096,1024),
            tr.nn.ReLU(True),
            tr.nn.Dropout(0.2),
            tr.nn.Linear(1024,num_class),
        )
    def forward(self, x):
        batchsize = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batchsize,-1)
        x = self.linear(x)
        return x

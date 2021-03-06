import torch as tr

class CnnModel(tr.nn.Module):
    def __init__(self, num_class=8):
        super(CnnModel, self).__init__()
        # data (40,368)
        self.conv1 = tr.nn.Sequential(
            tr.nn.Conv2d(1, 64, 3, 1),  # (38, 366)
            tr.nn.ReLU(True),
            tr.nn.Conv2d(64, 64, 3, 1),  # (36, 364)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(2),  # (18, 182)
            tr.nn.BatchNorm2d(64),
        )
        self.conv2 = tr.nn.Sequential(
            tr.nn.Conv2d(64, 128, 3, 1, 1),  # (18, 182)
            tr.nn.ReLU(True),
            tr.nn.Conv2d(128, 128, 3, 1),  # (16, 180)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(2),  # (8, 90)
            tr.nn.BatchNorm2d(128),
        )
        self.conv3 = tr.nn.Sequential(
            tr.nn.Conv2d(128, 128, 3, 1),  # (6, 88)
            tr.nn.ReLU(True),
            tr.nn.Conv2d(128, 128, 3, 1),  # (4, 86)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(2),  # (2, 43)
            tr.nn.BatchNorm2d(128),
        )
        self.linear = tr.nn.Sequential(
            tr.nn.Linear(2 * 43 * 128, 4096),
            tr.nn.ReLU(True),
            tr.nn.Dropout(0.2),
            tr.nn.Linear(4096, 1024),
            tr.nn.ReLU(True),
            tr.nn.Dropout(0.2),
            tr.nn.Linear(1024, num_class),
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batchsize, -1)
        x = self.linear(x)
        return x


class CRNNModel(tr.nn.Module):
    def __init__(self, num_class=8):
        super(CRNNModel, self).__init__()
        # data (40,368)
        self.conv1 = tr.nn.Sequential(
            tr.nn.Conv2d(1, 64, 3, 1),  # (38, 366)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(2),  # (19, 183)
            tr.nn.BatchNorm2d(64),
        )
        self.conv2 = tr.nn.Sequential(
            tr.nn.Conv2d(64, 128, 3, 1, 1),  # (19, 183)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(kernel_size=(2, 3)),  # (9, 91)
            tr.nn.BatchNorm2d(128),
        )
        self.conv3 = tr.nn.Sequential(
            tr.nn.Conv2d(128, 256, 3, 1, 1),  # (9, 91)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(kernel_size=(3, 4)),  # (4, 45)
            tr.nn.BatchNorm2d(256),
        )
        self.conv4 = tr.nn.Sequential(
            tr.nn.Conv2d(256, 256, 3, 1, 1),  # (2, 43)
            tr.nn.ReLU(True),
            tr.nn.MaxPool2d(kernel_size=(3, 4)),  # (1, 21)
            tr.nn.BatchNorm2d(256),
        )
        # input (256, 1, 21) --> (21, 256)
        self.rnn = tr.nn.GRU(input_size=256, hidden_size=256, dropout=0.4, bidirectional=True, num_layers=2)
        self.linear = tr.nn.Linear(in_features=256 * 2, out_features=num_class)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(batchsize, 256, -1)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)[0]
        x = x[-1, :, :]
        x = self.linear(x)
        return x



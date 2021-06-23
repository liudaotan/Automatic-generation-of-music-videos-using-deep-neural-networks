import torch as tr
from torch.utils.data import Dataset, DataLoader
import Utils.MusicDataset
import CNNmodels.CNN as CNN
import CNNmodels.CRNN as CRNN
import matplotlib.pyplot as plt
import numpy as np

# initialize the dataset
music_data = Utils.MusicDataset.MusicDataLoader("processed_data/processed_data.npy")
# seperate the dataset into training set and test set
trainset_size = int(music_data.__len__() * 0.85)
testset_size = music_data.__len__() - trainset_size
train_set, test_set = tr.utils.data.random_split(music_data, [trainset_size, testset_size])
# initialize the dataloader
train_loader = DataLoader(train_set, batch_size=120, num_workers=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=50, num_workers=11, pin_memory=True)

epochs = 3
lr = 0.0001
print_iters = 50
# model = CNN.CnnModel(num_class=8).cuda()
model = CRNN.CRNNModel(num_class=8).cuda()
criterion = tr.nn.BCEWithLogitsLoss()
optimizer = tr.optim.Adam(model.parameters(), lr=lr)
loss_list = []
tolerance = 0.0005
predecssing_loss = 0.0
run = True

for e in range(epochs):
    sum_loss = 0.0
    # if not run:
    #     break
    print('------------------------->epoch: ', e)
    for _, item in enumerate(train_loader):
        data, label = item
        y_ = model(data.cuda())
        loss = criterion(y_, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
        if _ % print_iters == (print_iters - 1):
            loss_list.append(sum_loss.item()/ print_iters)
            print("------> training loss:", sum_loss.item() / print_iters)
            # if np.abs(sum_loss.item() / print_iters - predecssing_loss) < tolerance:
            #     run = False
            #     break
            # else:
            #     predecssing_loss = sum_loss.item() / print_iters
            sum_loss = 0.0

plt.plot(loss_list)
plt.show()

# tr.save(model.state_dict(), "CNNmodels/cnnModel2.pth")
tr.save(model.state_dict(), "CNNmodels/rcnnModel1.pth")
model.eval()
sum_loss = 0.0
correct = 0.0
model.cuda()

with tr.no_grad():
    for _, item in enumerate(test_loader):
        data, label = item
        label = label.cuda()
        y_ = model(data.cuda())
        pred = tr.nn.functional.sigmoid(y_)
        loss = criterion(y_, label)
        sum_loss += loss

        for idx in range(data.shape[0]):
            if tr.equal((tr.nn.functional.sigmoid(y_[idx]) > 0.5).to(tr.float), label[idx]):
                correct += 1
    acc = correct / len(test_set)
    test_loss = sum_loss / _
    print("test accuracy:", acc)


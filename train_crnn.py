import torch as tr
from torch.utils.data import Dataset, DataLoader
import utils.MusicDataset as MusicDataset
import mymodels.MusicGenresModels as genre_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix as mcm

# initialize the dataset
music_data = MusicDataset.MusicDataLoader("data/audio/processed_data/processed_data.npy")
# seperate the dataset into training set and test set
trainset_size = int(music_data.__len__() * 0.85)
testset_size = music_data.__len__() - trainset_size
train_set, test_set = tr.utils.data.random_split(music_data, [trainset_size, testset_size])
# initialize the dataloader
train_loader = DataLoader(train_set, batch_size=120, num_workers=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=50, num_workers=11, pin_memory=True)

epochs = 10
lr = 0.0001
print_iters = 50
# model = CNN.CnnModel(num_class=8).cuda()
model = genre_model.CRNNModel(num_class=8).cuda()
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

# tr.save(model.state_dict(), "Models/cnnModel2.trained_model")
tr.save(model.state_dict(), "resources/trained_model/crnnModel1.pth")
model.eval()
sum_loss = 0.0
correct = 0.0
model.cuda()
classes= ['Experimental', 'Electronic', 'Rock', 'Instrumental', 'Folk', 'Pop', 'Hip-Hop','International']

multilabel_pred_list = []
multilabel_label_list = []

with tr.no_grad():
    for _, item in enumerate(test_loader):
        data, label = item
        label = label.cuda()
        y_ = model(data.cuda())
        loss = criterion(y_, label)
        sum_loss += loss
        for idx in range(data.shape[0]):
            if tr.equal((tr.sigmoid(y_[idx]) > 0.5).to(tr.float), label[idx]):
                correct += 1
        multilabel_pred_list.extend((tr.sigmoid(y_) > 0.5).to(tr.int).cpu().tolist())
        multilabel_label_list.extend(label.cpu().tolist())
    acc = correct / len(test_set)
    test_loss = sum_loss / _

    print("test accuracy:", acc)
mcm_res = mcm(y_true=multilabel_label_list, y_pred=multilabel_pred_list)
for _, item in enumerate(mcm_res):
    print(classes[_])
    print(item)
    print("-??-??-??-??-??-??-??-??-")


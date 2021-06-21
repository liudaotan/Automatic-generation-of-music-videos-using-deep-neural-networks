import torch as tr
from torch.utils.data import Dataset, DataLoader
import Utils.MusicDataset
import CNNmodels.Model as Model
import matplotlib.pyplot as plt

# initialize the dataset
music_data = Utils.MusicDataset.MusicDataLoader("processed_data/processed_data.npy")

# initialize the dataloader
test_loader = DataLoader(music_data, batch_size=80, num_workers=10, shuffle=True)

model = Model.CnnModel(num_class=8).cuda()
criterion = tr.nn.BCEWithLogitsLoss()
model_path = "CNNmodels/cnnModel2.pth"

model.load_state_dict(tr.load(model_path))
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
    acc = correct / len(music_data)
    test_loss = sum_loss / _
    print("test accuracy:", acc)

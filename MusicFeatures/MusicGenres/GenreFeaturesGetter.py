from MusicFeatures.MusicGenres.Models import Models
import torch
import torchaudio
import torchvision


chunk_size = 10

def loadmodel():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    pthpath = 'MusicGenres/crnnModel1.pth'
    print("loading the model.......")
    model = Models.CRNNModel()
    model.load_state_dict(torch.load(pthpath, map_location=device))
    return model


def prediction(model, data):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()
    predict_list = []
    y_list = []
    sum_y_ = torch.zeros(1, 8)
    num_segment = len(data)
    with torch.no_grad():
        for item in data:
            item = item.unsqueeze(0)
            y_ = model(item.to(device))
            predict_list.append((torch.sigmoid(y_) > 0.5).view(-1).nonzero().tolist())
            y_list.append(y_)
            sum_y_ += y_.cpu()
            sum_y_ /= item.shape[1] // num_segment

        y_tensor = torch.stack(y_list, axis=0)
        res_norm = torchvision.transforms.Normalize((0.5,), (0.5,))
        y_tensor = res_norm(y_tensor)
        overall_genre = torch.mean(y_tensor, dim=0)
        # overall_genre = (torch.nn.functional.sigmoid(y_tensor) > 0.5).view(-1).nonzero().tolist()
        # overall_genre = torch.sum(y_tensor, dim=0)
    return predict_list, overall_genre, y_tensor


# y,sr = torchaudio.load("prototype.mp3")
# model = loadmodel()
# mfcc = mfcc_preprocessing((y,sr))
# prediction(model, mfcc)

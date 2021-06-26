from Utils.MFCCgenerator import mfcc_preprocessing
import Models.Models as Models
import torch
import torchaudio
import torchvision


def loadmodel():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    pthpath = 'Models/crnnModel1.pth'
    print("loading the model.......")
    model = Models.CRNNModel()
    model.load_state_dict(torch.load(pthpath,map_location=device))
    return model


def prediction(model, data):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
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
        y_tensor = torch.mean(y_tensor, dim=0)
        # overall_genre = (torch.nn.functional.sigmoid(y_tensor) > 0.5).view(-1).nonzero().tolist()
        overall_genre = torch.sum(y_tensor, dim=0)
    return predict_list, overall_genre


def genre_features_getter(signal, sr=44100, frame_ms=100):
    signal_len = len(signal)
    signal_len_sec = signal_len // sr if signal_len % sr == 0 else signal_len // sr + 1
    repeat_per_sec = 1000 / frame_ms
    mfccs_chunks = mfcc_preprocessing((signal, sr))

def preprocessing(filepath):
    audio, fs = torchaudio.load(filepath)
    if audio.dim() > 1:
        data = torch.sum(audio, axis=0).unsqueeze(0)
    else:
        data = audio.unsqueeze(0)
    mfccs = mfcc_preprocessing((data, fs), 10, train=False)
    return mfccs
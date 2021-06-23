import sys
import getopt
import torch
import torchaudio
import torchvision
import CNNmodels.CRNN as Model
import Utils.MFCCgenerator as mfccgenerator

def help_msg():
    print('usage:')
    print('-h, --help: print help message.')
    print('-f, --file: aduio file')


def main(argv):
    filepath = ""

    if len(argv) <= 1:
        help_msg()
        sys.exit(0)

    try:
        opts, args = getopt.getopt(argv[1:], 'hf:', ['file='])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    for o, a in opts:
        if o in ('-h', '--help'):
            help_msg()
            sys.exit(1)
        elif o in ('-f', '--file:'):
            filepath = a
        else:
            print('unhandled option')
            sys.exit(3)
    return filepath


def preprocessing(filepath):
    audio, fs = torchaudio.load(filepath)
    sample_rate = 44100
    if audio.dim() > 1:
        data = torch.sum(audio, axis=0).unsqueeze(0)
    else:
        data = audio.unsqueeze(0)
    # MFCC_tranfromer = torchaudio.transforms.MFCC(sample_rate=sample_rate, log_mels=True,
    #                                              melkwargs={'n_fft': 1200, 'win_length': 1200,
    #                                                         'normalized': True})
    # MFCC_Norm = torchvision.transforms.Normalize((0.5,), (0.5,))
    # data = torch.nn.functional.pad(data, (
    #     0, data.shape[1] // (sample_rate * 5) * sample_rate * 5 + 1200 - data.shape[1]))
    # mfccs = MFCC_tranfromer(data)
    # mfccs = MFCC_Norm(mfccs.unsqueeze(0))
    mfccs = mfccgenerator.mfcc_preprocessing((data, fs), 10,train=False)
    return mfccs


def loadmodel():
    pthpath = 'CNNmodels/rcnnModel1.pth'
    print("loading the model.......")
    model = Model.CRNNModel()
    model.load_state_dict(torch.load(pthpath))
    return model


def prediction(model, data):
    model.cuda()
    model.eval()
    predict_list = []
    y_list = []
    sum_y_ = torch.zeros(1, 8)
    num_segment = len(data)
    with torch.no_grad():
        for item in data:
            item = item.unsqueeze(0)
            y_ = model(item.cuda())
            predict_list.append((torch.nn.functional.sigmoid(y_) > 0.5).view(-1).nonzero().tolist())
            y_list.append(y_)
            sum_y_ += y_.cpu()
            sum_y_ /= item.shape[1] // num_segment

        y_tensor = torch.stack(y_list,axis=0)
        res_norm = torchvision.transforms.Normalize((0.5,), (0.5,))
        y_tensor = res_norm(y_tensor)
        y_tensor = torch.mean(y_tensor, dim=0)
        overall_genre = (torch.nn.functional.sigmoid(y_tensor) > 0.5).view(-1).nonzero().tolist()
    return predict_list, overall_genre


def showres(predict_list, overall_genre):
    class_dict = {0: 'Experimental', 1: 'Electronic', 2: 'Rock', 3: 'Instrumental', 4: 'Folk', 5: 'Pop', 6: 'Hip-Hop',
                  7: 'International'}
    for idx, item in enumerate(predict_list):
        classes = [class_dict[cat[0]] for cat in item]
        if len(classes) == 0:
            classes = 'unknown'
        print("From seconds ", idx * 10, " to ", (idx + 1) * 10, " the music is ", classes)
    print("------------overall-------------")
    overall_classes = class_dict[overall_genre[0][0]] if len(overall_genre)>0 else 'unknown'
    print("The genre of the music is", overall_classes)


if __name__ == '__main__':
    filepath = main(sys.argv)
    # ----------load the model------------
    model = loadmodel()
    # ----------preprocessing------------
    mfccs_data = preprocessing(filepath)
    # ----------prediction------------
    predict_list, overall_genre = prediction(model, mfccs_data)
    # ----------show result------------
    showres(predict_list, overall_genre)

import sys
import getopt
import torch
import GenreFeaturesGetter as genregetter
import torch.nn.functional as F
from Utils.MFCCgenerator import mfcc_preprocessing
import torchaudio

def help_msg():
    print('usage:')
    print('-h, --help: print help message.')
    print('-f, --file: audio file')


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


def showres(predict_list, overall_genre):
    class_dict = {0: 'Experimental', 1: 'Electronic', 2: 'Rock', 3: 'Instrumental', 4: 'Folk', 5: 'Pop', 6: 'Hip-Hop',
                  7: 'International'}
    for idx, item in enumerate(predict_list):
        classes = [class_dict[cat[0]] for cat in item]
        if len(classes) == 0:
            classes = 'unknown'
        print("From seconds ", idx * 10, " to ", (idx + 1) * 10, " the music is ", classes)
    print("------------overall-------------")
    # overall_classes = class_dict[overall_genre[0][0]] if len(overall_genre)>0 else 'unknown'
    # print("The genre of the music is", overall_classes)
    genre_prob = F.softmax(overall_genre)
    top5_genre_prob, top5_genre_name = torch.topk(genre_prob, 5)
    top5_genre_prob = top5_genre_prob.squeeze()
    top5_genre_name = top5_genre_name.squeeze()
    print("genres prob:",
          {class_dict[top5_genre_name[i].item()]: round(top5_genre_prob[i].item(), 2) for i in range(5)})

def preprocessing(filepath):
    signal,sr = torchaudio.load(filepath)
    mfccs_chunks = mfcc_preprocessing((signal,sr ), train=False)
    return mfccs_chunks

if __name__ == '__main__':
    filepath = main(sys.argv)
    # ----------load the model------------
    model = genregetter.loadmodel()
    # ----------preprocessing------------
    mfccs_data = preprocessing(filepath)
    # ----------prediction------------
    predict_list, overall_genre, y_tensor, = genregetter.prediction(model, mfccs_data)
    # ----------show result------------
    showres(predict_list, overall_genre)

import sys
import getopt
import os
import torch as tr
from features.FeaturesLoader import FeaturesLoader


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


def loadGenerator(model, device, para_file_path="resource/trained_model/dcgan_generator"):
    model.load_state_dict(tr.load(para_file_path, map_location=device))
    model.to(device)
    return model


def featuresGetter(model, audio_file_path, para_file_path="resource/trained_model/crnnModel1.trained_model"):
    features_loader = FeaturesLoader(torch_model=model, para_file_path=para_file_path)
    features = features_loader.getFeatures(audio_file_path)
    return features


if __name__ == '__main__':
    print("None")

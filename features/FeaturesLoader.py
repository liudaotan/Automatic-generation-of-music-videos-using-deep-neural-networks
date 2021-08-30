import torch
import torchaudio
import torch.nn.functional as F

import mymodels
import torchvision
from features.Features import FeaturesGenerator


def loadmodel(model, para_file_path):
    """ This function load parameters to the model

    Parameters
    ----------
    model: torch.nn.Module
        the module which is used to extract features

    para_file_path: sting
        the model parameters file's path.

    Return
    ------
    model: torch.nn.Module

    """
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("loading the model.......")
    model.load_state_dict(torch.load(para_file_path, map_location=device))
    return model


def prediction(model, data):
    """ predict the music genre

    Parameter
    ---------
    model: torch.nn.Module
        The model used to predict the music genre

    data: list
        a list of MFCCs chunks

    Return
    ------
    predict_list: list
        genres of MFCCs chunks, the element of the list is the index of the genres(see 'class_list' in 'utils.config').

    overall_genre: torch.tensor (shape: (1 * num_classes))
        overall_genre presents the genre of the whole MFCCs, it is a probability vector for all classes.

    y_tensor: torch.tensor (shape: (num_chunks * num_classes))
        all predictions of MFCCs chunks will be stored in y_tensor, these predictions are outputs of the model.

    """
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
            category = (torch.sigmoid(y_) > 0.5).view(-1).nonzero().tolist()
            if len(category) == 0:
                category = [[8]]
            predict_list.append(category)
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


class FeaturesLoader:
    """
    This class provides the timbre and genres features,
    also, it can return a concatenated tensor of both timbre and genres features.
    """

    def __init__(self, torch_model, para_file_path, frame_len=0.1):
        """
        Parameters
        ----------
        torch_model: torch.nn.Module
            the model which is used to extract features

        para_file_path: sting
            the model parameters file's path.
        """
        # the length of a chunk, the unit of this argument is second
        # which means chunk_size=10 stands for that each chunk accounts for 10 seconds.
        self.chunk_size = 10
        # the length of a frame. the unit of this argument is second.
        self.frame_len = frame_len
        self.model = loadmodel(torch_model, para_file_path)
        self.features_generator = FeaturesGenerator()

    def getTimbreFeatures(self, audio, sr, ):
        """
        Parameters
        ----------

        Return
        ------
        """
        spec_centroid = self.features_generator.spectral_centroid(audio, sr, window_len=int(self.frame_len * 2 * sr),
                                                                  hop_len=int(self.frame_len * sr),
                                                                  n_fft=int(self.frame_len * 2 * sr))
        spec_centroid = F.normalize(spec_centroid, dim=0)
        return spec_centroid

    def getGenresFeatures(self, audio, sr, model, frame_len=0.1):
        """
        Parameters
        ----------
        audio: torch.tensor (shape(num_channels * len_audio))
            the signal extracted from the music. The signal should be a dual-channel signal or a single channel signal.

        sr: int
            sample rate. It typically is 44100.

        model: torch.nn.Module
            the model which is used to extract features

        frame_len: float
            the length of a frame. The unit of this argument is second. (Default: 0.1 second)

        Return
        ------
        res: torch.tensor
            genres of MFCCs chunks. The genre vector repeat (chunk_size / frame_len) times to make the genres tensor
            align the timbre features.

        """
        repeated_list = []

        if audio.dim() > 1:
            audio = torch.sum(audio, axis=0).unsqueeze(0)
        else:
            audio = audio.unsqueeze(0)

        mfccs = self.features_generator.mfcc_preprocessing((audio, sr), self.chunk_size, train=False)
        predict_list, overall_genre, y_list = prediction(model, mfccs)
        for item in y_list:
            item = torch.softmax(item, dim=1)
            repeated_item = item.repeat(int(self.chunk_size / frame_len), 1)
            repeated_list.append(repeated_item)
        res = torch.cat(repeated_list, dim=0)
        overall_genre = torch.softmax(overall_genre, dim=1)
        repeated_overall_genre = overall_genre.repeat(int(self.chunk_size / frame_len), 1)
        res = torch.cat((res, repeated_overall_genre), dim=0)
        return [item[0][0] for item in predict_list], res

    def loadMusic(self, path):
        """
        Parameters
        ----------
        path: string
            the path of the audio loaded in this module.

        Return
        ------
        audio: torch.tensor (shape(num_channels * len_audio))
            the signal extracted from the music. The signal should be a dual-channel signal or a single channel signal.

        sr: int
            sample rate. It typically is 44100.

        """
        audio, sr = torchaudio.load(path)

        return audio, sr

    def getFeatures(self, path):
        """
        Parameters
        ----------
        path: string
            the path of the audio loaded in this module.

        Return
        ------
        features: torch.tensor (shape (num_chunks * (num_timbre_features + num_genres)))
            the result is a tensor that consists of timbre features and genres features. The timbre features are
            concatenated before the genre's features. So, if the number of the timbre features is 2, the first two
            indexes of dimension 1 of the result refer to the timbre features.

        """
        audio, sr = self.loadMusic(path)
        genre, genre_vectors = self.getGenresFeatures(audio, sr, self.model, frame_len=self.frame_len)
        timbre = self.getTimbreFeatures(audio, sr).view(-1, 1)
        features_len = timbre.shape[0]
        features = torch.cat((timbre, genre_vectors[:features_len, :].cpu()), dim=1)
        return genre, features


if __name__ == '__main__':
    features_loader = FeaturesLoader(mymodels.CRNNModel(), para_file_path='../resources/trained_model/crnnModel1.pth',
                                     frame_len=0.025)
    genre, audio_features = features_loader.getFeatures("../resources/music/exciting.wav")
    print(audio_features.shape)

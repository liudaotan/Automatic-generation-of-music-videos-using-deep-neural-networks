from MusicGenres import GenreFeaturesGetter as genrefeaturesgetter
from FeaturesExtraction import Timbre as timbrefeaturesgetter
from MusicGenres.Utils import MFCCgenerator as mfccgenerator
import torch
import torchaudio
import torch.nn.functional as F

class FeaturesGenerator:
    def __init__(self):
        self.chunk_size = 10
        self.frame_len = 0.1
        self.model = genrefeaturesgetter.loadmodel()

    def getTimbreFeatures(self, audio, sr, ):
        spec_centroid = timbrefeaturesgetter.spectral_centroid(audio, sr, window_len=int(self.frame_len * 2 * sr),
                                                               hop_len=int(self.frame_len * sr),
                                                               n_fft=int(self.frame_len * 2 * sr))
        spec_centroid = F.normalize(spec_centroid,dim=0)
        return spec_centroid

    def getGenresFeatures(self, audio, sr, model, frame_len=0.1):
        repeated_list = []

        if audio.dim() > 1:
            audio = torch.sum(audio, axis=0).unsqueeze(0)
        else:
            audio = audio.unsqueeze(0)

        mfccs = mfccgenerator.mfcc_preprocessing((audio, sr), self.chunk_size, train=False)
        predict_list, overall_genre, y_list = genrefeaturesgetter.prediction(model, mfccs)
        for item in y_list:
            item = torch.softmax(item, dim=1)
            repeated_item = item.repeat(int(self.chunk_size / frame_len), 1)
            repeated_list.append(repeated_item)
        res = torch.cat(repeated_list, dim=0)
        overall_genre = torch.softmax(overall_genre, dim=1)
        repeated_overall_genre = overall_genre.repeat(int(self.chunk_size / frame_len), 1)
        res = torch.cat((res, repeated_overall_genre), dim=0)
        return res

    def loadMusic(self, path):
        audio, sr = torchaudio.load(path)
        return audio, sr

    def getFeatures(self, path):
        audio, sr = self.loadMusic(path)
        genre = self.getGenresFeatures(audio, sr, self.model)
        timbre = self.getTimbreFeatures(audio, sr).view(-1,1)
        features_len = timbre.shape[0]
        features = torch.cat((timbre, genre[:features_len,:].cpu()), dim=1)
        return features

# featuresgenerator = FeaturesGenerator()
# features = featuresgenerator.getFeatures("MusicGenres/prototype.mp3")

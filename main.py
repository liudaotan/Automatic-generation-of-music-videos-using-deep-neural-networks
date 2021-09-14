import math
import shutil

import numpy
import torch
import librosa
import torchaudio
from features.FeaturesLoader import FeaturesLoader
import numpy as np
import matplotlib.pyplot as plt
import ffmpeg
import mymodels.MusicGenresModels as music_genre_models
import scipy.special
import mymodels.Gan_structure as gan_model
import os
from argparse import ArgumentParser


def mel_norm_freq_filter_clip(y, sr, hop_len, filter_list, n_mels=128, clip_min=-1, clip_max=1):
    """
    Parameters:
    -----------
    y: numpy.array
        signal

    sr: int
        sample rate

    hop_len: int
        hop length

    filter_list:
        the indices of mel bin that are chosen.

    n_mels: int
        the number of mel bin.

    Return:
    ------
    """
    # # obatin the mel-spectrogram
    # mel = librosa.feature.melspectrogram(y, sr, n_mels=n_mels, hop_length=hop_len)
    # # filter the mel spectrogram, only choose the mel bins we are interested in.
    # mfcc_perc_select = mel[filter_list[0], :]
    # mfcc_perc_select = (mfcc_perc_select - np.min(mfcc_perc_select))/ (np.max(mfcc_perc_select) - np.min(mfcc_perc_select))
    # # computer the average magnitude along frames.
    # mel_norm = np.mean(mfcc_perc_select, axis=0)
    # return mel_norm

    mel = librosa.feature.melspectrogram(y, sr, n_mels=n_mels, hop_length=hop_len)
    # mel = np.log(mel + 1e-9)
    # filter the mel spectrogram, only choose the mel bins we are interested in.
    mel_perc_select = mel[filter_list[0], :]
    mel_perc_select = np.mean(mel_perc_select, axis=0)
    # standardize the mel_spectrogram
    mel_perc_select = np.log(mel_perc_select + 1e-9)
    mel_norm = (mel_perc_select - np.min(mel_perc_select))/(np.max(mel_perc_select) - np.min(mel_perc_select))
    return mel_norm


def magnitude_scaling(x, p=2):
    # the power operation scales values and makes values close to 0 or 1
    # x = x ** p
    # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    return np.exp(p*(x-np.mean(x)))


def freq2mel(freq):
    # convert frequency to mel-frequency
    mel_freq = 2595 * np.log10(1 + freq / 700.0)
    return mel_freq


def percussive_range(min_freq=None, max_freq=None, n_mels=128, bass_drum=True, snare_drum=True, crash_cymbals=True):
    # obtain mel bins of percussive components frequencies
    min_freq = 0.0 if min_freq is None else min_freq
    max_freq = 22050 or max_freq

    mel_min_freq = freq2mel(min_freq)
    mel_max_freq = freq2mel(max_freq)
    mel_range = np.linspace(mel_min_freq, mel_max_freq, num=n_mels)
    # the frequencies range of different kinds of instruments. These instruments usually work on heavy beats.
    crash_cymbals_freq_range = [3000, 5000]
    bass_drum_freq_range = [60, 100]
    snare_drum_freq_range = [120, 250]
    # choose instruments we want
    percussion_freq_collection = np.array([bass_drum_freq_range, snare_drum_freq_range, crash_cymbals_freq_range])
    percussion_type_idx = np.array([bass_drum, snare_drum, crash_cymbals]).astype("bool")
    choosing_freq_range = percussion_freq_collection[percussion_type_idx]
    # convert the frequency range to mel bin
    range_idx = lambda min, max: range(np.where(mel_range >= freq2mel(min))[0][0] - 1,
                                       np.where(mel_range <= freq2mel(max))[0][-1] + 1)
    percussive_freq_range = [list(range_idx(item[0], item[1])) for item in choosing_freq_range]
    return set().union(*percussive_freq_range)


class BaseVideoGenerator(object):
    """
    PGAN + beats detection

    The shape of PGAN's latent vector is n*512 which n is the number of pictures.
    """

    def __init__(self, frame_len=0.025, sec_per_keypic=7, gan_model=None, latent_dim=None):
        use_gpu = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
        # trained on high-quality celebrity faces "celebA" dataset
        # this model outputs 512 x 512 pixel images
        if gan_model is None:
            self.gan_model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                                            'PGAN', model_name='celebAHQ-512',
                                            pretrained=True, useGPU=use_gpu)
        else:
            self.gan_model = gan_model
        self.audio_model = music_genre_models.CRNNModel()
        self.features_loader = FeaturesLoader(torch_model=self.audio_model,
                                              para_file_path="resources/trained_model/crnnModel1.pth",
                                              frame_len=frame_len)
        # dimension of the latent vector
        if latent_dim is None:
            self.latent_dim = 512
        else:
            self.latent_dim = latent_dim
        # frames per second
        self.fps = math.ceil(1 // frame_len)
        # seconds per key picture
        self.sec_per_keypic = sec_per_keypic
        self.num_keypic = 0
        self.latent_features = False
        # emphasize weight
        self.emphasize_weight = 0.3
        self.impulse_win_len = 8
        self.impulse_win = torch.hann_window(self.impulse_win_len).view(-1, 1).to(self.device)
        self.frame_len = frame_len
        self.latent_features_is_init = False
        self.sample_rate = 44100
        self.is_pretrained = True if gan_model is None else False
        self.chunk_size=10

    @classmethod
    def load_audio_librosa(cls, file_path):
        return librosa.load(file_path)

    @classmethod
    def load_audio_torch(cls, file_path):
        return torchaudio.load(file_path)

    @classmethod
    def beat_detector(cls, signal, sr, hop_length):
        tempo, beats = librosa.beat.beat_track(y=signal, sr=sr, hop_length=hop_length)
        return beats

    def keyframe_init(self):
        num_cat = len(set(self.genre))
        cat_dict = dict(zip(set(self.genre), np.arange(len(set(self.genre))),))
        if hasattr(self.gan_model, 'buildNoiseData'):
            keypic, _ = self.gan_model.buildNoiseData(num_cat)
        else:
            keypic = torch.randn(num_cat, self.latent_dim, 1, 1).to(self.device)
            keypic = keypic.squeeze()
        keypic_temp = torch.zeros(self.num_keypic, self.latent_dim)
        for idx in range(self.num_keypic):
            cat_idx = idx * self.sec_per_keypic // self.chunk_size
            keypic_temp[idx] = keypic[cat_dict[self.genre[cat_idx]]]
        return keypic_temp

    def init_latent_vectors(self, file_path):
        # load features
        self.genre, features = self.features_loader.getFeatures(file_path)
        # load music by librosa
        signal_librosa, sr_librosa = librosa.load(file_path, sr=self.sample_rate)
        # obtain beats
        beats = self.beat_detector(signal_librosa, sr_librosa, math.ceil(self.frame_len * sr_librosa))
        # let spectral centroid follow probabilistic density
        spec_cent = features[:, 0]
        num_frames = features.shape[0]
        self.num_keypic = math.ceil(num_frames // (self.fps * self.sec_per_keypic))
        if hasattr(self.gan_model, 'buildNoiseData'):
            keypic, _ = self.gan_model.buildNoiseData(self.num_keypic)
        else:
            keypic = torch.randn(self.num_keypic, self.latent_dim, 1, 1).to(self.device)
            keypic = keypic.squeeze()
        #keypic = self.keyframe_init().to(self.device)
        # fps of a block
        fps_block = self.fps * self.sec_per_keypic
        self.latent_features = torch.zeros(fps_block * self.num_keypic, self.latent_dim).to(self.device)

        # init keyframes
        for i in range(self.num_keypic):
            self.latent_features[i * fps_block] = keypic[i]

        # init frames between two keyframes
        for i in range(self.num_keypic - 1):
            diff_vec = keypic[i + 1] - keypic[i]

            # get the spectral centroid probability
            spec_cent_partial = spec_cent[i * fps_block:(i + 1) * fps_block]

            # let spectral centroids follow probabilistic density
            spec_cent_partial = torch.cumsum(torch.softmax(spec_cent_partial.reshape(-1, 1), dim=0), dim=0).to(
                self.device)

            # get beats in this block
            beats_block = beats[(beats > i * fps_block) & (beats < (i + 1) * fps_block)] % fps_block

            # apply the impulse window to the probabilistic density of spectral
            # centroid to emphasize signals on beats
            half_impulse_win_len = int(self.impulse_win_len / 2)
            for beat in beats_block:
                if beat - half_impulse_win_len > 0 and beat + half_impulse_win_len < fps_block:
                    spec_cent_partial[beat - half_impulse_win_len: beat + half_impulse_win_len] += spec_cent_partial[
                                                                                                   beat - half_impulse_win_len: beat + half_impulse_win_len] * self.impulse_win * self.emphasize_weight
                elif beat - half_impulse_win_len <= 0:
                    spec_cent_partial[0: beat + half_impulse_win_len] += spec_cent_partial[
                                                                         0: beat + half_impulse_win_len] * self.impulse_win[
                                                                                                           :beat + half_impulse_win_len] * self.emphasize_weight
                else:
                    spec_cent_partial[beat - half_impulse_win_len:] += spec_cent_partial[
                                                                       beat - half_impulse_win_len:] * self.impulse_win[
                                                                                                       -(
                                                                                                               fps_block - beat + half_impulse_win_len):] * self.emphasize_weight

            # multiply the difference vector to the spectral centroid probabilistic density
            self.latent_features[i * fps_block:(i + 1) * fps_block] = torch.mul(spec_cent_partial,
                                                                                diff_vec.view(1, -1)) + \
                                                                      keypic[i]
        self.latent_features_is_init = True

    def generate_pictures(self, folder):
        folder_path = 'resources/imgs/' + folder
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
        else:
            os.mkdir(folder_path)
        if self.latent_features_is_init:
            for _, vec in enumerate(self.latent_features):
                with torch.no_grad():
                    if self.is_pretrained:
                        picture = self.gan_model.test(
                            vec.squeeze().unsqueeze(0).cuda()).squeeze().detach().cpu().permute(1,
                                                                                                2,
                                                                                                0).numpy()
                    else:
                        picture = self.gan_model(
                            vec.unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()).squeeze().detach().cpu().permute(1,
                                                                                                                2,
                                                                                                                0).numpy()
                    picture = (picture - np.min(picture)) / (np.max(picture) - np.min(picture))
                    plt.imsave('resources/imgs/' + folder + '/img%d.jpg' % _, picture)
        else:
            raise Exception("latent features haven't been initialized")

    def generate_video(self, save_folder, audio_path, picture_style, combined_method, verbose=True):
        input_imgs_path = 'resources/imgs/' + save_folder + '/img%d.jpg'
        input_video_path = 'resources/music_videos/' + save_folder + '.mp4'
        output_video_path = 'resources/music_videos/' + save_folder + '_' + picture_style + '_' + combined_method + '.mp4'
        # create video
        ffmpeg.input(input_imgs_path, framerate=self.fps).output(input_video_path).run()
        shutil.rmtree('resources/imgs/' + save_folder)
        # merge video and music
        input_video = ffmpeg.input(input_video_path)
        input_audio = ffmpeg.input(audio_path)
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_video_path).run()
        if verbose:
            print("video is successfully generated")

    def __call__(self, audio_path, picture_style, combined_method):
        filename = os.path.basename(audio_path).split(".")[0]
        print('--------------initializing latent vectors----------------')
        self.init_latent_vectors(audio_path)
        print('-----------------generating pictures---------------------')
        self.generate_pictures(filename)
        print('----------------generating the video---------------------')
        self.generate_video(filename, audio_path, picture_style, combined_method)
        os.remove("resources/music_videos/" + os.path.basename(audio_path).split(".")[0] + '.mp4')


class HpssVideoGenerator(BaseVideoGenerator):
    """
    This generator uses Median-filtering harmonic percussive source separation (HPSS)
    to extract the timbre features and maps features to the latent space
    """

    def __init__(self, **kwargs):
        """

        parameters
        ----------
        (see generate1.BaseVideoGenerator.__init__())

        frame_len: float
            the time of a frame(default 0.025 seconds)

        sec_per_keypic: int
            the time a key picture showing(default 7 seconds)

        """
        super(HpssVideoGenerator, self).__init__(**kwargs)
        self.emphasize_weight = 0.4

    @classmethod
    def get_hpss(cls, signal):
        # obtain harmonic and percussive component
        # harm: audio time series of the harmonic elements
        # perc: audio time series of the percussive elements
        harm, perc = librosa.effects.hpss(signal)
        return harm, perc

    def init_latent_vectors(self, file_path):
        # load features
        self.genre, features = self.features_loader.getFeatures(file_path)
        # load music by librosa
        # signal_librosa : audio time series ndarray
        # sr_librosa : sampling rate
        signal_librosa, sr_librosa = librosa.load(file_path, sr=self.sample_rate)
        hop_len = int(sr_librosa * self.frame_len)
        # obtain harmonic and percussive component
        harm, perc = self.get_hpss(signal_librosa)
        # using the bandstop filter to block frequencies from 500 to 3000Hz
        b, a = scipy.signal.butter(8, [500 * 2 / sr_librosa, 3000 * 2 / sr_librosa], btype='bandstop')
        perc = scipy.signal.filtfilt(b, a, perc)
        num_frames = features.shape[0]
        # let spectral centroid signal_librosa probabilistic density
        self.num_keypic = math.ceil(num_frames // (self.fps * self.sec_per_keypic))
        if hasattr(self.gan_model, 'buildNoiseData'):
            keypics, _ = self.gan_model.buildNoiseData(self.num_keypic)
        else:
            keypics = torch.randn(self.num_keypic, self.latent_dim).to(self.device)
        # keypics = self.keyframe_init().to(self.device)
        # fps of a block
        fps_block = self.fps * self.sec_per_keypic
        self.latent_features = torch.zeros(fps_block * self.num_keypic, self.latent_dim).to(self.device)

        # obtain the spectral centroid of harmonic component
        harm_spec_cent = librosa.feature.spectral_centroid(harm, sr=sr_librosa, hop_length=hop_len)
        # normalisation the harm_spec_cent
        harm_spec_cent_norm = (harm_spec_cent - np.min(harm_spec_cent)) / \
                              (np.max(harm_spec_cent) - np.min(harm_spec_cent))
        harm_spec_cent_norm = np.clip(harm_spec_cent_norm, a_min=1e-4, a_max=None)
        harm_spec_cent_norm = harm_spec_cent_norm.reshape(-1)

        # harm_spec = librosa.feature.melspectrogram(harm, sr=sr, n_mel=self.latent_dim, hop_length=hop_len)
        # perc_spec = librosa.feature.melspectrogram(perc, sr=sr_librosa, hop_length=hop_len)

        differ_matrix = torch.zeros(len(harm_spec_cent_norm), self.latent_dim).to(self.device)
        impulse_sign = (-1) ** numpy.random.randint(0, 2, len(harm_spec_cent_norm))

        # insert key pictures to key points.
        for i in range(self.num_keypic):
            self.latent_features[i * fps_block:(i + 1) * fps_block] = keypics[i]

        # compute the difference vectors
        for i in range(self.num_keypic - 1):
            differ_matrix[i * fps_block: (i + 1) * fps_block] = keypics[i + 1] - keypics[i]

        # expand the latent feature if the number of latent vectors is less than
        # size of the harmonic spectral centroid
        while len(harm_spec_cent_norm) > len(self.latent_features):
            self.latent_features = torch.cat((self.latent_features, self.latent_features[-1].view(1, -1)), axis=0)

        latent_features_diff_weight = np.zeros(len(harm_spec_cent_norm))
        latent_features_diff = np.zeros(len(harm_spec_cent_norm))

        # filter and normalize the percussive component's spectrogram
        percussive_component_mel_range = list(percussive_range())
        perc_spec_norm = mel_norm_freq_filter_clip(perc, sr_librosa, hop_len=hop_len,
                                                   filter_list=[percussive_component_mel_range])

        # apply the softmax to the normalized percussive component spectrogram
        # perc_spec_norm = scipy.special.softmax(perc_spec_norm)

        for i in range(self.num_keypic - 1):
            # interpolate vectors between two key frames by spectral centroid
            block_weight = harm_spec_cent_norm[fps_block * i: fps_block * (i + 1)]
            block_weight = (block_weight - np.min(block_weight)) / (np.max(block_weight) - np.min(block_weight))
            block_weight /= np.sum(block_weight)
            latent_features_diff[fps_block * i:fps_block * (i + 1)] = block_weight
            block_weight = np.cumsum(block_weight)
            latent_features_diff_weight[fps_block * i:fps_block * (i + 1)] = block_weight
        # emphasize frames corresponding to the percussive component
        perc_spec_norm = magnitude_scaling(perc_spec_norm)
        latent_features_diff_weight *= (1 + perc_spec_norm * self.emphasize_weight)
        latent_features_diff_weight = torch.tensor(latent_features_diff_weight, device=self.device).view(-1, 1)
        self.latent_features += latent_features_diff_weight * differ_matrix
        self.latent_features_is_init = True


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='Auto music video generator')
    parser.add_argument('--audio', type=str, help='audio file name, please put your audio file to the folder \'resources\\music\'')
    parser.add_argument('--model', type=str, default='landscape',
                        help='the model used to generate the video '
                             '(option: \'landscape\', \'abstract\', \'pretty_face\', \'face512\')')
    parser.add_argument('--method', type=str, default='base',
                        help='the method applied to the change of video (option: \'base\' or \'hpss\')')
    parser.add_argument('--emphasize', type=float, default=0.3,
                        help='the magnitude applied to the generator to control the strength of audio\'s change')
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    print('audio:', args.audio)
    print('model:', args.model)
    print('method:', args.method)
    music = args.audio
    picture_style = {'face512': 'PretrainedHighResolutionFace', 'landscape': 'landscape', 'pretty_face': 'pretty_face',
                     'abstract': 'abstractArt'}[args.model]
    combined_method = {'base': 'Base', 'hpss': 'Hpss'}[args.method]
    dc_generator_path = 'resources/trained_model/' + picture_style + '/DCGAN.pth'
    sr_generator_path = 'resources/trained_model/' + picture_style + '/SRGAN.pth'
    music_path = "resources/music/" + music
    if picture_style == 'PretrainedHighResolutionFace':
        if combined_method == 'Hpss':
            base_video_gen = HpssVideoGenerator()
        else:
            base_video_gen = BaseVideoGenerator()
        base_video_gen(music_path, 'faces512', combined_method)
    else:
        model_gen = gan_model.GAN_Generators(base_pth=dc_generator_path, boost_pth=sr_generator_path)
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        model_gen.to(device)

        if combined_method == 'Hpss':
            base_video_gen = HpssVideoGenerator(gan_model=model_gen, latent_dim=100)
        else:
            base_video_gen = BaseVideoGenerator(gan_model=model_gen, latent_dim=100)
        base_video_gen(music_path, picture_style, combined_method)


if __name__ == '__main__':
    main()

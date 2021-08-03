import math
import shutil

import numpy
import torch
import librosa
import torchaudio
from torchaudio import transforms as audio_transforms
from features.FeaturesLoader import FeaturesLoader
import matplotlib.pyplot as plt
import numpy as np
import ffmpeg
import os
import mymodels.MusicGenresModels as music_genre_models
import scipy.special
import mymodels.GANModel as gan_model


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
                                              para_file_path="resources/pth/crnnModel1.pth",
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

    def init_latent_vectors(self, file_path):
        # load features
        features = self.features_loader.getFeatures(file_path)
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
            keypic = torch.randn(self.num_keypic, self.latent_dim).to(self.device)
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
                        picture = self.gan_model.test(vec.squeeze().unsqueeze(0).cuda()).squeeze().detach().cpu().permute(1,
                                                                                                                          2,
                                                                                                                          0).numpy()
                    else:
                        picture = self.gan_model(vec.unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()).squeeze().detach().cpu().permute(1,
                                                                                                                            2,
                                                                                                                            0).numpy()
                    picture = (picture - np.min(picture)) / (np.max(picture) - np.min(picture))
                    plt.imsave('resources/imgs/' + folder + '/img%d.jpg' % _, picture)
        else:
            raise Exception("latent features haven't been initialized")

    def generate_video(self, save_folder, audio_path, verbose=True):
        input_imgs_path = 'resources/imgs/' + save_folder + '/img%d.jpg'
        input_video_path = 'resources/videos/' + save_folder + '.mp4'
        output_video_path = 'resources/videos/' + save_folder + '_full.mp4'
        # create video
        ffmpeg.input(input_imgs_path, framerate=self.fps).output(input_video_path).run()
        shutil.rmtree('resources/imgs/' + save_folder)
        # merge video and music
        input_video = ffmpeg.input(input_video_path)
        input_audio = ffmpeg.input(audio_path)
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_video_path).run()
        if verbose:
            print("video is successfully generated")

    def __call__(self, audio_path):
        filename = os.path.basename(audio_path).split(".")[0]
        print('--------------initializing latent vectors----------------')
        self.init_latent_vectors(audio_path)
        print('-----------------generating pictures---------------------')
        self.generate_pictures(filename)
        print('----------------generating the video---------------------')
        self.generate_video(filename, audio_path)


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
        self.emphasize_weight = 0.7
    @classmethod
    def get_hpss(cls, signal):
        harm, perc = librosa.effects.hpss(signal)
        return harm, perc

    def init_latent_vectors(self, file_path):
        # load features
        features = self.features_loader.getFeatures(file_path)
        # load music by librosa
        signal_librosa, sr_librosa = librosa.load(file_path, sr=self.sample_rate)
        hop_len = int(sr_librosa * self.frame_len)
        # obtain harmonic and percussive component
        harm, perc = self.get_hpss(signal_librosa)
        # let spectral centroid signal_librosa probabilistic density
        spec_cent = features[:, 0]
        num_frames = features.shape[0]
        self.num_keypic = math.ceil(num_frames // (self.fps * self.sec_per_keypic))
        if hasattr(self.gan_model, 'buildNoiseData'):
            keypics, _ = self.gan_model.buildNoiseData(self.num_keypic)
        else:
            keypics = torch.randn(self.num_keypic, self.latent_dim).to(self.device)
        # fps of a block
        fps_block = self.fps * self.sec_per_keypic
        self.latent_features = torch.zeros(fps_block * self.num_keypic, self.latent_dim).to(self.device)

        # harm_spec = librosa.feature.melspectrogram(harm, sr=sr, n_mel=self.latent_dim, hop_length=hop_len)
        perc_spec = librosa.feature.melspectrogram(perc, sr=sr_librosa, hop_length=hop_len)

        differ_matrix = torch.zeros(perc_spec.shape[1], self.latent_dim).to(self.device)
        impulse_sign = (-1)**numpy.random.randint(0, 2, (perc_spec.shape[1]))
        # insert key pictures to key points.
        for i in range(self.num_keypic):
            self.latent_features[i * fps_block:(i + 1) * fps_block] = keypics[i]

        # compute the difference vectors
        for i in range(self.num_keypic - 1):
            differ_matrix[i * fps_block: (i + 1) * fps_block] = keypics[i + 1] - keypics[i]

        while perc_spec.shape[1] > len(self.latent_features):
            self.latent_features = torch.cat((self.latent_features, self.latent_features[-1].view(1, -1)), axis=0)

        latent_features_diff_weight = np.zeros(perc_spec.shape[1])
        # obtain the spectral centroid of harmonic component
        harm_spec_cent = librosa.feature.spectral_centroid(harm, sr=sr_librosa, hop_length=hop_len)
        harm_spec_cent_norm = (harm_spec_cent - np.min(harm_spec_cent)) / \
                              (np.max(harm_spec_cent) - np.min(harm_spec_cent))
        harm_spec_cent_norm = np.clip(harm_spec_cent_norm, a_min=1e-4, a_max=None)
        harm_spec_cent_norm = harm_spec_cent_norm.reshape(-1)

        # normalize the percussive component's spectrogram
        perc_spec_mean = np.mean(perc_spec, axis=0)
        perc_spec_norm = (perc_spec_mean - np.min(perc_spec_mean)) / np.ptp(perc_spec_mean)
        # apply the softmax to the normalized percussive component spectrogram
        # perc_spec_norm = scipy.special.softmax(perc_spec_norm)

        for i in range(self.num_keypic - 1):
            # interpolate vectors between two key frames by spectral centroid
            block_weight = harm_spec_cent_norm[fps_block * i: fps_block * (i + 1)]
            block_weight = (block_weight - np.min(block_weight)) / (np.max(block_weight) - np.min(block_weight))
            block_weight /= np.sum(block_weight)
            block_weight = np.cumsum(block_weight)
            latent_features_diff_weight[fps_block * i:fps_block * (i + 1)] = block_weight
        # emphasize frames corresponding to the percussive component
        latent_features_diff_weight *= (1 + perc_spec_norm * self.emphasize_weight * impulse_sign)
        latent_features_diff_weight = torch.tensor(latent_features_diff_weight, device=self.device).view(-1, 1)
        self.latent_features += latent_features_diff_weight * differ_matrix
        self.latent_features_is_init = True


if __name__ == '__main__':
    dc_generator_path = 'resources/pth/landscape/netG_200_size64.pth'
    sr_generator_path = 'resources/pth/landscape/netG_3form_size64_to_size256.pth'
    model_gen = gan_model.Abstract_Generator(base_pth=dc_generator_path, boost_pth=sr_generator_path)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model_gen.to(device)
    # base_video_gen = BaseVideoGenerator(gan_model=model_gen, latent_dim=100)
    base_video_gen = HpssVideoGenerator(gan_model=model_gen, latent_dim=100)
    # base_video_gen = BaseVideoGenerator()
    # base_video_gen = HpssVideoGenerator()
    path = "resources/music/Metallica - NOTHING ELSE MATTERS.flac"
    base_video_gen(path)

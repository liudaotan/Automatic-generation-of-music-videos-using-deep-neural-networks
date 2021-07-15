import math
import shutil

import torch
import librosa
import torchaudio
from features.FeaturesLoader import FeaturesLoader
import mymodels.Models as Models
import matplotlib.pyplot as plt
import numpy as np
import ffmpeg
import os
import mymodels.Models as Models


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
        self.audio_model = Models.CRNNModel()
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
        self.emphasize_weight = 0.2
        self.impulse_win_len = 6
        self.impulse_win = torch.hann_window(self.impulse_win_len).view(-1, 1).to(self.device)
        self.frame_len = frame_len
        self.latent_features_is_init = False

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
        signal_librosa, sr_librosa = librosa.load(file_path)
        # obtain beats
        beats = self.beat_detector(signal_librosa, sr_librosa, math.ceil(self.frame_len * sr_librosa))
        # let spectral centroid follow probabilistic density
        spec_cent = features[:, 0]
        num_frames = features.shape[0]
        self.num_keypic = math.ceil(num_frames // (self.fps * self.sec_per_keypic))
        if hasattr(self.gan_model, 'buildNoiseData'):
            keypic, _ = self.gan_model.buildNoiseData(self.num_keypic)
        else:
            keypic = torch.randn(self.num_keypic, self.latent_dim)
        # fps of a block
        fps_bloc = self.fps * self.sec_per_keypic
        self.latent_features = torch.zeros(fps_bloc * self.num_keypic, self.latent_dim).to(self.device)

        # init keyframes
        for i in range(self.num_keypic):
            self.latent_features[i * fps_bloc] = keypic[i]

        # init frames between two keyframes
        for i in range(self.num_keypic - 1):
            diff_vec = keypic[i + 1] - keypic[i]

            # get the spectral centroid probability
            spec_cent_partial = spec_cent[i * fps_bloc:(i + 1) * fps_bloc]

            # let spectral centroids follow probabilistic density
            spec_cent_partial = torch.cumsum(torch.softmax(spec_cent_partial.reshape(-1, 1), dim=0), dim=0).to(
                self.device)

            # get beats in this block
            beats_block = beats[(beats > i * fps_bloc) & (beats < (i + 1) * fps_bloc)] % fps_bloc

            # apply the impulse window to the probabilistic density of spectral
            # centroid to emphasize signals on beats
            half_impulse_win_len = int(self.impulse_win_len / 2)
            for beat in beats_block:
                if beat - half_impulse_win_len > 0 and beat + half_impulse_win_len < fps_bloc:
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
                                                                                                               fps_bloc - beat + half_impulse_win_len):] * self.emphasize_weight

            # multiply the difference vector to the spectral centroid probabilistic density
            self.latent_features[i * fps_bloc:(i + 1) * fps_bloc] = torch.mul(spec_cent_partial, diff_vec.view(1, -1)) + \
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
                    # picture = self.gan_model.test(vec.squeeze().unsqueeze(0).cuda()).squeeze().detach().cpu().permute(1,
                    #                                                                                                   2,
                    #                                                                                                   0).numpy()
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

    @classmethod
    def get_hpss(cls, signal):
        harm, perc = librosa.decompose.hpss(s=signal)
        return harm, perc

    def init_latent_vectors(self, file_path):
        # load features
        features = self.features_loader.getFeatures(file_path)
        # load music by librosa
        signal_librosa, sr_librosa = librosa.load(file_path)
        # obtain harmonic and percussive component
        signal, sr = self.load_audio_librosa(file_path)
        harm, perc = self.get_hpss(signal)
        # let spectral centroid follow probabilistic density
        spec_cent = features[:, 0]
        num_frames = features.shape[0]
        self.num_keypic = math.ceil(num_frames // (self.fps * self.sec_per_keypic))
        keypic, _ = self.gan_model.buildNoiseData(self.num_keypic)
        # fps of a block
        fps_bloc = self.fps * self.sec_per_keypic
        self.latent_features = torch.zeros(fps_bloc * self.num_keypic, self.latent_dim).to(self.device)


if __name__ == '__main__':
    generator_path = 'resources/pth/netG_200_size64.pth'
    model_gen = Models.Generator(ngpu=1)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model_gen.load_state_dict(torch.load(generator_path, map_location=device))
    base_video_gen = BaseVideoGenerator(gan_model=model_gen, latent_dim=100)
    base_video_gen('resources/music/bj_new.mp3')

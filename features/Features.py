import torch as tr
import numpy as np
import torchaudio
import torch.nn.functional as F
import torchvision
import nnAudio
import nnAudio.Spectrogram


class FeaturesGenerator:

    # data preprocessing
    def mfcc_preprocessing(self, raw_signal, window_size_sec=10, chunk=True, train=True):
        """
        The data will be padded by zero and then processed by MFCC. The MFcc will be evenly sliced into many chunks by t
        ime distributed, it makes the model focus on the data in a short period of time.

        Parameters:
        ----------
        raw_signal:  set(tensor: signal, int: sample_rate)
            raw signal data extract by Librosa or TorchAudio, which is a set.

        window_size_sec: int
            the length of a window per second.

        chunk:  boolean('True' or 'False')
            slice the MFCCs if the 'chunk' is True, otherwise return MFCCs without any processing.

        train: boolean('True' or 'False')
            if it is true, the function will set the size of signal to 30s.

        Return
        ------
        MFCCs chunks: set([tensor, tensor ... ])
            a set of chunked mfcc data .

        """
        # sr = Config.sr
        # zero_padding_len = Config.zero_padding_len
        # music_len_sec = Config.music_len_sec
        # nfft = Config.nfft
        # fft_win_len = Config.fft_win_len

        sr = 44100
        zero_padding_len = 1800
        music_len_sec = 30
        nfft = 1200
        fft_win_len = 1200

        # if the audio is dual-channel, we use the mean of audio signal.
        signal = raw_signal[0]
        signal = tr.mean(signal, axis=0)

        if not train:
            music_len_sec = raw_signal[0].shape[1] // sr

        num_chunk = music_len_sec // window_size_sec

        # if the length of the data is small than the required length (num_chunk * the number of frames of a chunk), the signal
        # will be padded by zero. On the contrary, the excessing data will be discarded.
        if signal.shape[0] < sr * music_len_sec + zero_padding_len:
            signal = F.pad(signal, (0, sr * music_len_sec + zero_padding_len - signal.shape[0]))
        elif signal.shape[0] > sr * music_len_sec + zero_padding_len:
            signal = signal[:sr * music_len_sec + zero_padding_len]

        # apply Mfcc
        MFCC_tranfromer = torchaudio.transforms.MFCC(sample_rate=sr, log_mels=True,
                                                     melkwargs={'n_fft': nfft, 'win_length': fft_win_len,
                                                                'normalized': True})

        # normalize the data
        MFCC_Norm = torchvision.transforms.Normalize((0.5,), (0.5,))
        mfccs = MFCC_tranfromer(signal)[:, :-1]
        mfccs = MFCC_Norm(mfccs.unsqueeze(0))

        # slice the data
        if chunk:
            chunk_mfccs = tr.chunk(mfccs, num_chunk, dim=2)
            return chunk_mfccs
        else:
            return mfccs

    def cqt(self, signal, sr):
        return nnAudio.Spectrogram.CQT2010(sr=sr)(signal)

    def fft_freq(self, n_fft, sr):
        return tr.linspace(0, float(sr) / 2, int(1 + n_fft // 2))

    def spectral_centroid(self, signal, sr, n_fft=2048, window_len=2048, hop_len=512):
        """
        Parameters:
        ----------
        signal: torch.tensor
            The shape of the signal should be (2, n) or (n) where 'n' is the size of the sampling frames.

        sr:  int
            Sampling rate

        n_fft: int
            Size of Fourier transform. (Default: 2048)

        Returns
        -------
        spectral centroid: torch.tensor
            It returns the spectral centroid of the given audio, which is shape of (1, num_samples).
            Details about num_samples are in the document of'nnAudio.Spectrogram.STFT'

        """
        # check the dimension of the sigal
        signal_dim = signal.dim()
        if signal_dim == 1:
            signal = signal
        elif signal_dim == 2:
            signal = tr.mean(signal, dim=0)
        else:
            raise ValueError('the number of channels must be 1 or 2')

        spectrogram = nnAudio.Spectrogram.STFT(sr=sr, n_fft=n_fft, win_length=window_len, hop_length=hop_len,
                                               output_format="Magnitude", verbose=False)(signal).squeeze()
        fft_frequencies = self.fft_freq(n_fft, sr / 2).view(-1, 1)
        norm_spectrogram = spectrogram / tr.sum(spectrogram, dim=0)
        return tr.sum(fft_frequencies * norm_spectrogram, dim=0)


if __name__ == '__main__':
    features_gen = FeaturesGenerator()
    mfcc_chunks = features_gen.mfcc_preprocessing(torchaudio.load("../resources/music/space_oddity.mp3"),chunk=False, train=False)
    print(mfcc_chunks.shape)

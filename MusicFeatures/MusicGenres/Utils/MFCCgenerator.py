import torch as tr
import numpy as np
import torchaudio
import torch.nn.functional as F
import torchvision

import os


# data preprocessing
def mfcc_preprocessing(raw_signal, window_size_sec, chunk=True, train=True):
    """
    The data will be padded by zero and then processed by MFCC. The MFcc will be evenly sliced into many chunks by t
    ime distributed, it makes the model focus on the data in a short period of time.

    :param train:
    :param raw_signal: raw signal data extract by Librosa or TorchAudio, which is a set (tensor: signal, int: sample_rate).
    :param window_size_sec: the length of a window per second.
    :param chunk: slice the MFCCs if the 'chunk' is True, otherwise return MFCCs without any processing.
    :return: a set of chunked mfcc data ([tensor, tensor ... ]).
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

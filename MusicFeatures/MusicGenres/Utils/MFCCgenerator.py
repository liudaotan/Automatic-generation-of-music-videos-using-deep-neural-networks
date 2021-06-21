import torch as tr
import numpy as np
import torchaudio
import torch.nn.functional as F
import torchvision
import os

# load data from pickle
top_10_onehot_data = np.load("../processed_data/top_10_onehot_data.npy", allow_pickle=True)
idx_dict_list = np.load("../processed_data/idx_dict_list.npy", allow_pickle=True)
idx_dict_list = dict(idx_dict_list.tolist())
raw_idx2genres = idx_dict_list["raw_idx2genres"]
idx2genres = idx_dict_list["idx2genres"]
rawidx2newidx = idx_dict_list["rawidx2newidx"]


# data preprocessing
def mfcc_preprocessing(raw_signal, window_size_sec):

    """
    The data will be padded by zero and then processed by MFCC. The MFcc will be evenly sliced into many chunks by t
    ime distributed, it makes the model focus on the data in a short period of time.

    :param raw_signal: raw signal data extract by Librosa or TorchAudio, which is a set (tensor: signal, int: sample_rate)
    :param window_size_sec: the length of a window per second
    :return: a set of chunked mfcc data ([tensor, tensor ... ])
    """
    sr = 44100
    zero_padding_len = 1800
    music_len_sec = 30
    nfft = 1200
    fft_win_len = 1200

    num_chunk = 30 // window_size_sec
    # if the audio is dual-channel, we use the mean of audio signal.
    signal = raw_signal[0]
    signal = tr.mean(signal, axis=0)

    # if the length of the data is small than the required length (num_chunk * the number of frames of a chunk), the signal
    # will be padded by zero. On the contrary, the excessing data will be discarded.
    if signal.shape[0] < sr * music_len_sec+zero_padding_len:
        signal = F.pad(signal, (0, sr * music_len_sec+zero_padding_len - signal.shape[0]))
    elif signal.shape[0] > sr * music_len_sec+zero_padding_len:
        signal = signal[:sr*music_len_sec+zero_padding_len]

    # apply Mfcc
    MFCC_tranfromer = torchaudio.transforms.MFCC(sample_rate=sr, log_mels=True,
                                                 melkwargs={'n_fft': nfft, 'win_length': fft_win_len, 'normalized': True})

    # normalize the data
    MFCC_Norm = torchvision.transforms.Normalize((0.5,), (0.5,))
    mfccs = MFCC_tranfromer(signal)[:,:-1]
    mfccs = MFCC_Norm(mfccs.unsqueeze(0))

    # slice the data
    chunk_mfccs = tr.chunk(mfccs, num_chunk, dim=2)
    return chunk_mfccs

# get all audio files' name.
root = '../fma_small'
file_name_list = []
folders = list(filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root)))
for folder in folders:
    files = os.listdir(os.path.join(root, folder))
    file_name_list.extend([os.path.join(root, folder, file) for file in files])

# preprocess the data
# the data structure of the processed_data will be
# [(chunked_mfcc, one_hot_label), (chunked_mfcc, one_hot_label)....]
processed_data = []
top_10_onehot_data = dict(top_10_onehot_data)
for file_name in file_name_list:
    track_id = int(os.path.basename(file_name).split(".")[0])
    raw_signal = torchaudio.load(file_name)
    chunk_mfccs = mfcc_preprocessing(raw_signal, 5)
    label = top_10_onehot_data[track_id]
    processed_data.extend([(chunk.numpy(), label) for chunk in chunk_mfccs])

np.save("../processed_data/processed_data", processed_data)

import torch as tr
import numpy as np
import torchaudio
import torch.nn.functional as F
import torchvision
import os

top_10_onehot_data = np.load("../processed_data/top_10_onehot_data.npy", allow_pickle=True)
idx_dict_list = np.load("../processed_data/idx_dict_list.npy", allow_pickle=True)
idx_dict_list = dict(idx_dict_list.tolist())
raw_idx2genres = idx_dict_list["raw_idx2genres"]
idx2genres = idx_dict_list["idx2genres"]
rawidx2newidx = idx_dict_list["rawidx2newidx"]


def mfcc_preprocessing(raw_signal, window_size_sec):
    num_chunk = 30 // window_size_sec
    sr = 44100
    signal = raw_signal[0]
    signal = tr.mean(signal, axis=0)
    if signal.shape[0] < 44100 * 30+1800:
        signal = F.pad(signal, (0, 44100 * 30+1800 - signal.shape[0]))
    elif signal.shape[0] > 44100 * 30+1800:
        signal = signal[:44100*30+1800]
    MFCC_tranfromer = torchaudio.transforms.MFCC(sample_rate=sr, log_mels=True,
                                                 melkwargs={'n_fft': 1200, 'win_length': 1200, 'normalized': True})
    MFCC_Norm = torchvision.transforms.Normalize((0.5,), (0.5,))
    mfccs = MFCC_tranfromer(signal)[:,:-1]
    mfccs = MFCC_Norm(mfccs.unsqueeze(0))
    chunk_mfccs = tr.chunk(mfccs, num_chunk, dim=2)
    return chunk_mfccs


root = '../fma_small'
file_name_list = []
folders = list(filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root)))
for folder in folders:
    files = os.listdir(os.path.join(root, folder))
    file_name_list.extend([os.path.join(root, folder, file) for file in files])

processed_data = []
top_10_onehot_data = dict(top_10_onehot_data)
for file_name in file_name_list:
    track_id = int(os.path.basename(file_name).split(".")[0])
    raw_signal = torchaudio.load(file_name)
    chunk_mfccs = mfcc_preprocessing(raw_signal, 5)
    label = top_10_onehot_data[track_id]
    processed_data.extend([(chunk.numpy(), label) for chunk in chunk_mfccs])

np.save("../processed_data/processed_data", processed_data)

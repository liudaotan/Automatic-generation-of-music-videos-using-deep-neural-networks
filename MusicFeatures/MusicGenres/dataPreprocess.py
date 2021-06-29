import numpy as np
import torchaudio
import os
from Utils import MFCCgenerator as mfccgenerator

# load data from pickle
top_10_onehot_data = np.load("../processed_data/top_10_onehot_data.npy", allow_pickle=True)
idx_dict_list = np.load("../processed_data/idx_dict_list.npy", allow_pickle=True)
idx_dict_list = dict(idx_dict_list.tolist())
raw_idx2genres = idx_dict_list["raw_idx2genres"]
idx2genres = idx_dict_list["idx2genres"]
rawidx2newidx = idx_dict_list["rawidx2newidx"]


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
    chunk_mfccs = mfccgenerator.mfcc_preprocessing(raw_signal, 10)
    label = top_10_onehot_data[track_id]
    processed_data.extend([(chunk.numpy(), label) for chunk in chunk_mfccs])

np.save("../processed_data/processed_data", processed_data)

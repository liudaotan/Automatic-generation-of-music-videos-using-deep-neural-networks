import nnAudio.Spectrogram
import torch.nn
import torchaudio
import matplotlib.pyplot as plt
import torch
import numpy as np
#def spectral_centroid(signal, sr):
import librosa
import librosa.display

def cqt(signal, sr):
    return nnAudio.Spectrogram.CQT2010(sr=sr)(signal)

# class Spectral_Centroid(torch.nn.Module):
signal, sr = torchaudio.load("../MusicGenres/prototype.mp3")
stft = nnAudio.Spectrogram.STFT( output_format='Magnitude')
signal = torch.sum(signal, dim=0).unsqueeze(0)
spectrogram = stft(signal)
img = spectrogram.squeeze().numpy()
librosa.display.specshow(img)
plt.show()

def spectral_centroid(signal, sr):


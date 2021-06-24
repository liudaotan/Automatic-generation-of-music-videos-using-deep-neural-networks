import nnAudio.Spectrogram
import torch.nn
import torch



def cqt(signal, sr):
    return nnAudio.Spectrogram.CQT2010(sr=sr)(signal)


def fft_freq(n_fft, sr):
    return torch.linspace(0, float(sr) / 2, int(1 + n_fft // 2))


def spectral_centroid(signal, sr, n_fft=2048):
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
        signal = torch.mean(signal, dim=0)
    else:
        raise ValueError('the number of channels must be 1 or 2')

    spectrogram = nnAudio.Spectrogram.STFT(sr=sr, n_fft=n_fft, output_format="Magnitude",verbose=False)(signal).squeeze()
    fft_frequencies = fft_freq(n_fft, sr/2).view(-1, 1)
    norm_spectrogram = spectrogram/torch.sum(spectrogram, dim=0)
    return torch.sum(fft_frequencies * norm_spectrogram, dim=0)







import librosa
import librosa.display
import numpy as np
import torch


def transform(waveform, n_fft=512, hop_length=160, n_mels=40):
    """
    Returns melspectrogram
    :param waveform: torch.Tensor. From torchaudio.datasets.VoxCeleb1Identification
    :param n_fft: int. window size of STFT calculation
    :param hop_length: int. STFT hop size of STFT calculation
    :param n_mels: int. number of mel scale frequencies to use
    :return: torch.Tensor
    """
    normalized_waveform = librosa.util.normalize(waveform)
    stft = librosa.core.stft(normalized_waveform, n_fft=n_fft, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(S=stft, n_mels=n_mels)
    mel_log = np.log(mel + 1e-9)
    mel_normalized = librosa.util.normalize(mel_log)
    # Remove phase
    mel_normalized_real = np.real(mel_normalized)
    tensor = torch.zeros([1, mel_normalized_real.shape[0], mel_normalized_real.shape[1]])
    tensor[0, :, :] = torch.Tensor(mel_normalized_real)
    return tensor

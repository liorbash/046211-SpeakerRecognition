import librosa
import librosa.display
import numpy as np

def transform(waveform, n_fft=512, hop_length=160, n_mels=40):
    normalizedy = librosa.util.normalize(waveform)
    stft = librosa.core.stft(normalizedy, n_fft=n_fft, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(S=stft, n_mels=n_mels)
    mellog = np.log(mel + 1e-9)
    melnormalized = librosa.util.normalize(mellog)
    # Remove phase
    melnormalized_real = np.real(melnormalized)
    tensor = torch.zeros([1, melnormalized_real.shape[0], melnormalized_real.shape[1]])
    tensor[0,:,:] = torch.Tensor(melnormalized_real)
    return tensor

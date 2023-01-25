from torchaudio import datasets
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MinMaxScaler
from data_utils import Melspectogram

VOXCELEB1_NUM_SPEAKERS = 1211
SPLIT_FILE = 'iden_split.txt'

class VoxCaleb1MelSpecDataset(Dataset):
    def __init__(self, data_dir, subset, transform=None, min_max_scale=False, normalize_mel=True, sr=16000, download=False):
        """
        :param data_dir: string. Directory with all the audio files
        :param subset: string. 'train', 'val'/'dev' or 'test'
        :param transform: torch.transforms. Input's transforms.
        :param min_max_scale: bool. If True, then MinMax scaling the melspectrogram.
        :param normalize_mel: bool. If True , then normalizing by frequency bins the melspectrogram.
        It can be only normalize_mel or only min_max_scale or neither. Where normalize_mel is first.
        :param sr: int. Audio files' sampling rate
        :param download: bool.
        """
        if data_dir.endswith('/'):
            self.data_dir = data_dir[:-1]
        else:
            self.data_dir = data_dir
        self.transform = transform
        self.min_max_scale = min_max_scale
        self.normalize_mel = normalize_mel
        if subset == 'val':
            subset = 'dev'
        self.dataset = datasets.VoxCeleb1Identification(root=self.data_dir, subset=subset, download=download)
        self.idx = {'waveform': 0, 'sr': 1, 'speaker_id': 2, 'path_wav': 3}
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]

        # Transforms
        sample = list(sample)
        if self.normalize_mel:
            sample[self.idx['waveform']] = Melspectogram.transform(sample[self.idx['waveform']].cpu().numpy()[0])

        if self.transform:
            sample[self.idx['waveform']] = self.transform(sample[self.idx['waveform']])

        if self.min_max_scale:
            scaler = MinMaxScaler()
            sample[self.idx['waveform']][0, :, :] = torch.Tensor(scaler.fit_transform(sample[self.idx['waveform']][0, :, :]))

        sample = tuple(sample)

        # RGB
        image = torch.cat((sample[self.idx['waveform']], sample[self.idx['waveform']], sample[self.idx['waveform']]), 0)
        # labels start at 0 instead of 1
        label = sample[self.idx['speaker_id']] - 1
        return image, label

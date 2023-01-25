import torchaudio.transforms
import torchvision.transforms
import torch

from data_utils.VoxCaleb1MelSpecDataset import VoxCaleb1MelSpecDataset


def get_datasets(input_size, dataset_dir='.', download=False, min_max_scale=False, normalize_mel=True):
    """
    Creates VoxCaleb1MelSpecDataset  instances for train, validation and test.
    :param input_size: int. for ResNet-18 is 224
    :param dataset_dir: str
    :param download: bool. If True, then downloading the dataset to dataset_dir.
    :param min_max_scale: bool. If True, then MinMax scaling the melspectrogram.
    :param normalize_mel: bool. If True , then normalizing by frequency bins the melspectrogram.
    It can be only normalize_mel or only min_max_scale or neither. Where normalize_mel is first.
    :return: dict of VoxCaleb1MelSpecDataset instances by keys: 'train', 'val' and 'test'
    """
    if normalize_mel:
        transforms = torchvision.transforms.Resize((input_size, input_size))
    else:
        transforms = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                 f_min=0.0, f_max=8000, pad=0, n_mels=40),
            torchvision.transforms.Resize((input_size, input_size))
        )
    train_dataset = VoxCaleb1MelSpecDataset(dataset_dir, 'train', transform=transforms, download=download,
                                            min_max_scale=min_max_scale, normalize_mel=normalize_mel)
    validation_dataset = VoxCaleb1MelSpecDataset(dataset_dir, 'dev', transform=transforms, download=download,
                                                 min_max_scale=min_max_scale, normalize_mel=normalize_mel)
    test_dataset = VoxCaleb1MelSpecDataset(dataset_dir, 'test', transform=transforms, download=download,
                                           min_max_scale=min_max_scale, normalize_mel=normalize_mel)
    return {'train': train_dataset, 'val': validation_dataset, 'test': test_dataset}


def get_dataloader(dataset, batch_size, num_workers=2, shuffle=True):
    """
    returns torch.utils.data.DataLoader instance
    :param dataset: VoxCaleb1MelSpecDataset
    :param batch_size: int
    :param num_workers: int
    :param shuffle: bool
    :return: torch.utils.data.DataLoader
    """
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=shuffle)

    return data_loader


def get_dataloaders(dataset_list, train_batch_size, num_workers=2, shuffle=True):
    """

    :param dataset_list: dict of VoxCaleb1MelSpecDataset with keys: 'train', 'val' and 'test'
    :param train_batch_size: int
    :param num_workers: int
    :param shuffle: bool
    :return: dict of torch.utils.data.DataLoader instances by keys: 'train', 'val' and 'test'
    """
    train_dataloader = get_dataloader(dataset_list['train'], train_batch_size, num_workers=num_workers,
                                      shuffle=shuffle)
    validation_dataloader = get_dataloader(dataset_list['val'], len(dataset_list['val']), num_workers=num_workers,
                                           shuffle=shuffle)
    test_dataloader = get_dataloader(dataset_list['test'], train_batch_size, num_workers=num_workers,
                                     shuffle=shuffle)
    return {'train': train_dataloader, 'val': validation_dataloader, 'test': test_dataloader}

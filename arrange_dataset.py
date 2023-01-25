import numpy as np
import os
import argparse
from torchaudio import datasets

VOXCELEB1_NUM_SPEAKERS = 1211
SPLIT_FILE = 'iden_split.txt'


def reduce_speakers(num_classes, data_dir):
    """
    Reduces the speakers in dataset to be from 1 to num_classes and re-splits the dataset to train, val and test if
    resplit is True. All that by changing the 'iden_split.txt' file.
    :param num_classes: int
    :param data_dir: string.
    :param resplit: bool. If True, then re-splitting the dataset to train, val and test according to train_size
                          and val_size
    :param train_size: float. Relevant if resplit is True
    :param val_size: float. Relevant if resplit is True
    :return:  None
    """
    if data_dir.endswith('/'):
        iden_split_path = data_dir[:-1]
    else:
        iden_split_path = data_dir
    iden_split_file = iden_split_path + '/' + SPLIT_FILE

    with open(iden_split_file, 'r') as f:
        iden_split_lines = f.readlines()

    # Rename original file
    os.rename(iden_split_file, f'{iden_split_path}/{SPLIT_FILE}_old_num_speakers.txt')

    new_iden_split = ''
    for line in iden_split_lines:
        # line example: "1 id10003/E_6MjfYr0sQ/00023.wav\n"
        speaker_id = int(line.split(' ')[1].split('/')[0][3:])
        if speaker_id <= num_classes:
            new_iden_split += line
    with open(iden_split_file, 'w') as f:
        f.write(new_iden_split)
    return


def split_dataset(data_dir, train_size=0.6, val_size=0.2):
    """
    Splits the dataset to train, validation and test according to train_size and val_size by changing the
    'iden_split.txt' file.
    :param data_dir: string.
    :param train_size: float
    :param val_size: float
    :return: None
    """
    if data_dir.endswith('/'):
        iden_split_path = data_dir[:-1]
    else:
        iden_split_path = data_dir

    iden_split_file = iden_split_path + '/' + SPLIT_FILE

    with open(iden_split_file, 'r') as f:
        iden_split_lines = f.readlines()

    # Rename original file
    os.rename(iden_split_file, f'{iden_split_path}/{SPLIT_FILE}_old_split.txt')

    n_samples = len(iden_split_lines)
    rand_gen = np.random.RandomState(0)
    # Generating a shuffled vector of indices
    indices = np.arange(n_samples)
    rand_gen.shuffle(indices)

    # Split the indices into 80% train (full) / 20% test
    n_samples_train = int(n_samples * train_size)
    n_samples_val = int(n_samples * val_size)
    train_indices = indices[:n_samples_train]
    val_indices = indices[n_samples_train:n_samples_train + n_samples_val]

    new_iden_split = ''
    phase_dict = {'train': 1, 'val': 2, 'test': 3}
    for i, line in enumerate(iden_split_lines):
        if i in train_indices:
            new_line = str(phase_dict['train']) + line[1:]
        elif i in val_indices:
            new_line = str(phase_dict['val']) + line[1:]
        else:
            new_line = str(phase_dict['test']) + line[1:]
        new_iden_split += new_line

    with open(iden_split_file, 'w') as f:
        f.write(new_iden_split)
    return


def main():
    parser = argparse.ArgumentParser(description='Arranges VoxCeleb1 dataset according to num_classes and '
                                                 're-splits the dataset if required')

    parser.add_argument('--n_speakers', action="store", dest="n_speakers", type=int,
                        help='the number of speakers wanted in the dataset, needs to be <= 1211')
    parser.add_argument('--download', action="store_true", dest="download", default=False,
                        help='whether to download the dataset')
    parser.add_argument('--dataset_dir', action="store", dest="dataset_dir", type=str,
                        help='path to the directory of the dataset')
    parser.add_argument('--resplit', action="store_true", dest="resplit", default=False,
                        help='if True, then re-splitting the dataset to train, val and test according to --train_size '
                             'and --val_size')
    parser.add_argument('--train_size', action="store", dest="train_size", type=float, default=0.6,
                        help="train's element of the dataset")
    parser.add_argument('--val_size', action="store", dest="val_size", type=float, default=0.2,
                        help="validation's element of the dataset")

    args = parser.parse_args()

    assert args.n_speakers <= VOXCELEB1_NUM_SPEAKERS
    assert (args.train_size + args.val_size) < 1

    if args.download:
        print('Downloading VoxCeleb1 dataset')
        datasets.VoxCeleb1Identification(root=args.dataset_dir, subset='train', download=True)

    if args.n_speakers < VOXCELEB1_NUM_SPEAKERS:
        print(f'Reducing to {args.n_speakers} speakers')
        reduce_speakers(args.n_speakers, args.dataset_dir)

    if args.resplit:
        print(f'Re-splitting dataset to {args.train_size} train, {args.val_size} validation and '
              f'{1 - args.train_size - args.val_size} test')
        split_dataset(args.dataset_dir, train_size=args.train_size, val_size=args.val_size)

    return


if __name__ == '__main__':
    main()

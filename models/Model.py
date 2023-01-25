import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import random

from data_utils import DataLoader
from trainer.Trainer import Trainer, initialize_resnet_model
from models.ContrastiveCenterLoss import ContrastiveCenterLoss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def train_cross_entropy(num_classes, device, dataset_dir='.', checkpoint_path='./drive/MyDrive/046211', batch_size=64,
                        num_epochs=20, num_workers=2, eps=1e-09, betas=(0.9, 0.98), step_size=7, gamma=0.1, lr=1e-4,
                        download_dataset=False):
    """
Train ResNet-18 on Speaker Recognition task with Cross-Entropy loss
    :param num_classes: int
    :param device: string. 'cpu' or 'cuda:X'
    :param dataset_dir: string. path where dataset is saved or would be downloaded
    :param checkpoint_path: string. path where checkpoints would be saved
    :param batch_size: int
    :param num_epochs: int
    :param num_workers: int
    for ResNet Adam optimizer:
    :param eps: float
    :param betas: tuple of float
    for lr scheduler:
    :param step_size: int
    :param gamma: float
    :param lr: float

    :param download_dataset: bool
    :return:
    """
    model_ft, input_size = initialize_resnet_model(num_classes=num_classes, feature_extract=False, use_pretrained=True)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, betas=betas, eps=eps)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    trainer = Trainer(model_ft, device, input_size, criterion, optimizer_ft, exp_lr_scheduler, batch_size=batch_size,
                      num_epochs=num_epochs, num_workers=num_workers, dataset_dir=dataset_dir,
                      download_dataset=download_dataset, checkpoint_path=checkpoint_path)

    # returns the checkpoint path of the best epoch according to validation accuracy
    checkpoint_path = trainer.train_model()
    checkpoint = torch.load(checkpoint_path)
    model_ft.load_state_dict(checkpoint['model_state_dict'])
    optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])

    eval_trainer = Trainer(model_ft, device, input_size, criterion, optimizer_ft, exp_lr_scheduler, batch_size=batch_size,
                      num_epochs=num_epochs, num_workers=num_workers, dataset_dir=dataset_dir,
                      download_dataset=download_dataset, checkpoint_path=checkpoint_path)
    eval_trainer.evaluate_model()
    return


def train_cross_entropy_and_contrastive_center_loss(num_classes, device, dataset_dir='.',
                                                    checkpoint_path='./drive/MyDrive/046211',batch_size=64, num_epochs=20,
                                                    num_workers=2, eps=1e-09, betas=(0.9, 0.98),
                                                    step_size=10, gamma=0.1, lr=1e-4,  optimizer_name_ccl='Adagrad',
                                                    lr_ccl=0.001, lambda_c=1.0, download_dataset=False):
    """
Train ResNet-18 on Speaker Recognition task with Cross-Entropy and Contrastive-center loss
    :param num_classes: int
    :param device: string. 'cpu' or 'cuda:X'
    :param dataset_dir: string. path where dataset is saved or would be downloaded
    :param checkpoint_path: string. path where checkpoints would be saved
    :param batch_size: int
    :param num_epochs: int
    :param num_workers: int
    for ResNet Adam optimizer:
    :param eps: float
    :param betas: tuple of float
    for lr scheduler:
    :param step_size: int
    :param gamma: float
    :param lr: float
    for Contrastive-center loss optimizer:
    :param optimizer_name_ccl:
    :param lr_ccl: float
    :param lambda_c: float

    :param download_dataset: bool
    :return: None
    """
    model_ft, input_size = initialize_resnet_model(num_classes=num_classes, feature_extract=False, use_pretrained=True)
    model_ft = model_ft.to(device)

    # Losses
    ce_loss = nn.CrossEntropyLoss()
    center_loss = ContrastiveCenterLoss(dim_hidden=512, device=device, num_classes=num_classes, lambda_c=lambda_c)
    criterion = [ce_loss, center_loss]

    # Optimizers & Scheduler
    optimizer_nn = optim.Adam(model_ft.parameters(), lr=lr, betas=betas, eps=eps)
    optimizer_ccl = getattr(optim, optimizer_name_ccl)(center_loss.parameters(), lr=lr_ccl)
    optimizer = [optimizer_nn, optimizer_ccl]

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_nn, step_size=step_size, gamma=gamma)

    trainer = Trainer(model_ft, device, input_size, criterion, optimizer, exp_lr_scheduler, batch_size=batch_size,
                      num_epochs=num_epochs, num_workers=num_workers, dataset_dir=dataset_dir,
                      download_dataset=download_dataset, checkpoint_path=checkpoint_path)

    # returns the checkpoint path of the best epoch according to validation accuracy
    best_checkpoint_path = trainer.train_model_ccl()
    checkpoint = torch.load(best_checkpoint_path)
    model_ft.load_state_dict(checkpoint['model_state_dict'])
    optimizer_nn.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer = [optimizer_nn, optimizer_ccl]

    eval_trainer = Trainer(model_ft, device, input_size, criterion, optimizer, exp_lr_scheduler, batch_size=batch_size,
                      num_epochs=num_epochs, num_workers=num_workers, dataset_dir=dataset_dir,
                      download_dataset=download_dataset, checkpoint_path=checkpoint_path)
    eval_trainer.evaluate_model()
    return
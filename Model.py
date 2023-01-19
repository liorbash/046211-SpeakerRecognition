import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import DataLoader
from Trainer import Trainer, initialize_resnet_model
from ContrastiveCenterLoss import ContrastiveCenterLoss

from google.colab import drive

drive.mount('/content/drive')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def train_cross_entropy(num_classes, device, dataset_dir='.', batch_size=64, num_epochs=20, num_workers=2,
                        eps=1e-09, step_size=7, gamma=0.1, lr=1e-4, betas=(0.9, 0.98), download_dataset=False):
    """
  Args:
      num_classes (int)
      device (string): 'cpu' or 'cuda:X'
      dataset_dir (string)
      batch_size  (int)
      num_epochs (int)
      num_workers (int)
      download_dataset (bool)
      for Adam optimizer:
        lr (float)
        betas (tuple)
        eps (float)
      for lr scheduler:
        step_size (int)
        gamma (float)
  """
    model_ft, input_size = initialize_resnet_model(num_classes=num_classes, feature_extract=False, use_pretrained=True)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, betas=betas, eps=eps)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    trainer = Trainer(model_ft, device, input_size, criterion, optimizer_ft, exp_lr_scheduler, batch_size=batch_size,
                      num_epochs=num_epochs, num_workers=num_workers, dataset_dir=dataset_dir,
                      download_dataset=download_dataset)
    trainer.train_model()
    return


def train_cross_entropy_and_contrastive_center_loss(num_classes, device, dataset_dir='.', batch_size=64, num_epochs=20,
                                                    num_workers=2, download_dataset=False, eps=1e-09, step_size=7,
                                                    gamma=0.1, lr=1e-4, betas=(0.9, 0.98), lr_ccl=0.001, lambda_c=1.0):
    """
  Args:
      num_classes (int)
      device (string): 'cpu' or 'cuda:X'
      dataset_dir (string)
      batch_size  (int)
      num_epochs (int)
      num_workers (int)
      download_dataset (bool)
      for Adam optimizer:
        lr (float)
        betas (tuple)
        eps (float)
      for lr scheduler:
        step_size (int)
        gamma (float)
      for contrastive center loss:
        lr_ccl (float)
        lambda_c (float): weight of contrastive center loss in criterion
  """
    model_ft, input_size = initialize_model(num_classes=num_classes, feature_extract=False, use_pretrained=True)
    model_ft = model_ft.to(device)

    # Losses
    ce_loss = nn.CrossEntropyLoss()
    center_loss = ContrastiveCenterLoss(dim_hidden=512, num_classes=num_classes, lambda_c=lambda_c)
    criterion = [ce_loss, center_loss]

    # Optimizers & Scheduler
    optimizer_nn = optim.Adam(resnet18_model_fine_tuning.parameters(), lr=lr, betas=betas, eps=eps)
    optimizer_ccl = optim.Adagrad(center_loss.parameters(), lr=lr_ccl)
    optimizer = [optimizer_nn, optimizer_ccl]

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_nn, step_size=step_size, gamma=gamma)

    trainer = Trainer(model_ft, device, input_size, criterion, optimizer, exp_lr_scheduler, batch_size=batch_size,
                      num_epochs=num_epochs, num_workers=num_workers, dataset_dir=dataset_dir,
                      download_dataset=download_dataset)
    trainer.train_model('train')
    return


if __name__ == '__main__':
    set_seed(0)
    # device - cpu or gpu?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

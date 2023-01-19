import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import copy
import time
import datetime
import pandas as pd

import DataLoader
import Metric


def set_parameter_requires_grad(model, feature_extracting=False):
    """
    Args:
        model (torch.model)
        feature_extracting (bool): If true, frozen model. else, model.parameters are requires grad
    """
    if feature_extracting:
        # frozen model
        for param in model.parameters():
            param.requires_grad = False
    else:
        # fine-tuning
        for param in model.parameters():
            param.requires_grad = True


def initialize_resnet_model(num_classes, feature_extract, use_pretrained=True):
    """
    Args:
        num_classes (int)
        feature_extract (bool): If true, frozen model. else, model.parameters are requires grad
        use_pretrained (bool)
    """
    weights = 'DEFAULT' if use_pretrained else None
    # to use other checkpoints than the default ones, check the model's available chekpoints here:
    # https://pytorch.org/vision/stable/models.html
    model_ft = models.resnet18(weights=weights)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
    input_size = 224
    return model_ft, input_size


class Trainer():
    def __init__(self, model, device, input_size, criterion, optimizer_ft, scheduler, batch_size, dataset_dir='.',
                 download_dataset=False, num_epochs=20, num_workers=2, min_max_scale=False, normalize_mel=True,
                 shuffle=True):
        """
        Args:
            model (torch.model)
            device (string): 'cpu' or 'cuda:X'
            input_size (int)
            criterion (nn)
            optimizer_ft (torch.optim)
            scheduler (torch.optim.lr_scheduler)
            batch_size (int)
            dataset_dir (str)
            download_dataset (bool)
            num_epochs (int)
            num_workers (int)
            min_max_scale (bool): If true, input is min-max scaled
            normalize_mel (bool): If true, normalize input by mean and std according to time
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer_ft
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.datasets = DataLoader.get_datasets(input_size, dataset_dir=dataset_dir, download=download_dataset,
                                                min_max_scale=min_max_scale, normalize_mel=normalize_mel)
        self.dataloaders = DataLoader.get_dataloaders(self.datasets, batch_size, num_workers=num_workers,
                                                      shuffle=shuffle)

    def train_model(self, phase):
        """
        Args:
            phase (string): 'train' or 'val'
        """
        since = time.time()
        start_datetime = str(datetime.datetime.now())

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        iterations_results = {'Iteration': [], 'Epoch': [], 'Loss': [], 'Accuracy': [], 'Phase': []}
        epochs_results = {'Epoch': [], 'Loss': [], 'Accuracy': [], 'Phase': []}

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for iter_i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)

                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    iter_loss = loss.item() * inputs.size(0)
                    running_loss += iter_loss
                    iter_corrects = torch.sum(preds == labels.data)
                    running_corrects += iter_corrects

                    # collect data
                    iterations_results['Iteration'].append(iter_i)
                    iterations_results['Epoch'].append(epoch)
                    iterations_results['Loss'].append(iter_loss)
                    iterations_results['Accuracy'].append(iter_corrects.cpu().numpy())
                    iterations_results['Phase'].append(phase)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(self.datasets[phase])
                epoch_acc = running_corrects.double() / len(self.datasets[phase])

                # collect data
                epochs_results['Epoch'].append(epoch)
                epochs_results['Loss'].append(epoch_loss)
                epochs_results['Accuracy'].append(epoch_acc.cpu().numpy())
                epochs_results['Phase'].append(phase)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    # Save checkpoint
                    save_path = f'./drive/MyDrive/046211/{start_datetime}_model_epoch{str(epoch)}.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': epoch_loss,
                    }, save_path)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # save data to excels
        epochs_df = pd.DataFrame(epochs_results)
        iterations_df = pd.DataFrame(iterations_results)
        model_df = pd.DataFrame(
            {'name': ['criterion', 'optimizer', 'scheduler.step_size', 'scheduler.gamma', 'batch_size'],
             'value': [str(self.criterion), str(self.optimizer), str(self.scheduler.step_size),
                       str(self.scheduler.gamma), str(self.batch_size)]})
        with pd.ExcelWriter(f'./drive/MyDrive/046211/{start_datetime}_model_results.xlsx') as writer:
            model_df.to_excel(writer, sheet_name='Model', index=False)
            epochs_df.to_excel(writer, sheet_name='Epochs', index=False)
            iterations_df.to_excel(writer, sheet_name='Iterations', index=False)
        return

    def train_model_ccl(self, phase):
        since = time.time()
        start_datetime = str(datetime.datetime.now())

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        iterations_results = {'Iteration': [], 'Epoch': [], 'Loss': [], 'Accuracy': [], 'Phase': []}
        epochs_results = {'Epoch': [], 'Loss': [], 'Accuracy': [], 'Phase': []}

        optimizer_nn, optimizer_ccl = self.optimizer
        criterion_nn, criterion_ccl = self.criterion

        # Using average pooling layer output as input to contrastive center loss
        return_nodes = {
            "avgpool": "avgpool"
        }
        avgpool = create_feature_extractor(self.model, return_nodes=return_nodes)

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for iter_i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer_nn.zero_grad()
                    optimizer_ccl.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        avgpool_outputs = avgpool(inputs)['avgpool']

                        _, preds = torch.max(outputs, 1)
                        loss = criterion_nn(outputs, labels) + criterion_ccl(labels, avgpool_outputs)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer_nn.step()
                            optimizer_ccl.step()

                    # statistics
                    iter_loss = iter_loss = loss.item() * inputs.size(0)
                    running_loss += iter_loss
                    iter_corrects = torch.sum(preds == labels.data)
                    running_corrects += iter_corrects

                    # collect data
                    iterations_results['Iteration'].append(iter_i)
                    iterations_results['Epoch'].append(epoch)
                    iterations_results['Loss'].append(iter_loss)
                    iterations_results['Accuracy'].append(iter_corrects.cpu().numpy())
                    iterations_results['Phase'].append(phase)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(self.datasets[phase])
                epoch_acc = running_corrects.double() / len(self.datasets[phase])

                # collect data
                epochs_results['Epoch'].append(epoch)
                epochs_results['Loss'].append(epoch_loss)
                epochs_results['Accuracy'].append(epoch_acc.cpu().numpy())
                epochs_results['Phase'].append(phase)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    # Save checkpoint
                    save_path_nn = f'./drive/MyDrive/046211/{start_datetime}_model_epoch{str(epoch)}_nn.pt'
                    save_path_ccl = f'./drive/MyDrive/046211/{start_datetime}_model_epoch{str(epoch)}_ccl.pt'
                    # Save the NN optimizer
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer_nn.state_dict(),
                        'loss': epoch_loss,
                    }, save_path_nn)
                    # Save the CCL optimizer
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer_ccl.state_dict(),
                        'loss': epoch_loss,
                    }, save_path_ccl)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # save data to excels
        epochs_df = pd.DataFrame(epochs_results)
        iterations_df = pd.DataFrame(iterations_results)
        model_df = pd.DataFrame({'name': ['criterion', 'optimizer_nn', 'optimizer_ccl', 'scheduler.step_size',
                                          'scheduler.gamma', 'batch_size'],
                                 'value': [str(self.criterion), str(optimizer_nn), str(optimizer_ccl),
                                           str(self.scheduler.step_size), str(self.scheduler.gamma),
                                           str(self.batch_size)]})
        with pd.ExcelWriter(f'./drive/MyDrive/046211/{start_datetime}_model_results.xlsx') as writer:
            model_df.to_excel(writer, sheet_name='Model', index=False)
            epochs_df.to_excel(writer, sheet_name='Epochs', index=False)
            iterations_df.to_excel(writer, sheet_name='Iterations', index=False)
        return

    def evaluate_model(self):
        X_test, y_test = list(iter(self.dataloaders['test']))[0]
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test.to(self.device))
            _, preds = torch.max(test_outputs, 1)
        test_acc = Metric.accurarcy(preds, y_test)
        test_top_1_acc = Metric.top_k_acc(preds, y_test, k=1)
        test_top_5_acc = Metric.top_k_acc(preds, y_test, k=5)
        print(f'Test accurarcy: {test_acc}\nTest top 1% accurarcy: {test_top_1_acc}\n'
              f'Test top 5% accurarcy: {test_top_5_acc}')
        return

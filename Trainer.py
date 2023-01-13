import torch
import torch.nn as nn
from torchvision import models
import copy
import time

import DataLoader

def set_parameter_requires_grad(model, feature_extracting=False):
  if feature_extracting:
    # frozen model
    for param in model.parameters():
      param.requires_grad = False
  else:
    # fine-tuning
    for param in model.parameters():
      param.requires_grad = True
  # note: you can also mix between frozen layers and trainable layers, but you'll need a custom
  # function that loops over the model's layers and you specify which layers are frozen.

def set_layers_requires_grad(model, num_layers_to_freeze):
  count = 0
  for child in model.children():
    count += 1
    if count < num_layers_to_freeze:
      for param in child.parameters():
        param.requires_grad = False
    else:
      for param in model.parameters():
        param.requires_grad = True
        
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
  # Initialize these variables which will be set in this if statement. Each of these
  # variables is model specific.
  model_ft = None
  input_size = 0 # image size, e.g. (3, 224, 224)
  # new method from torchvision >= 0.13
  weights = 'DEFAULT' if use_pretrained else None
  # to use other checkpoints than the default ones, check the model's available chekpoints here:
  # https://pytorch.org/vision/stable/models.html
  if model_name == "resnet":
    """ 
    Resnet18
    """
    # new method from torchvision >= 0.13
    model_ft = models.resnet18(weights=weights)
    set_parameter_requires_grad(model_ft, feature_extract)
    #set_layers_requires_grad(model_ft, num_layers_to_freeze)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes) # replace the last FC layer
    input_size = 224
  elif model_name == "alexnet":
    """ 
    Alexnet
    """
    # new method from torchvision >= 0.13
    model_ft = models.alexnet(weights=weights)
    set_parameter_requires_grad(model_ft, feature_extract)
    #set_layers_requires_grad(model_ft, num_layers_to_freeze)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224
  elif model_name == "vgg":
    """ 
    VGG16
    """
    # new method from torchvision >= 0.13
    model_ft = models.vgg16(weights=weights)
    set_parameter_requires_grad(model_ft, feature_extract)
    #set_layers_requires_grad(model_ft, num_layers_to_freeze)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224
  else:
    raise NotImplementedError
  return model_ft, input_size
  
class Trainer():
  def __init__(self, model, device, input_size, criterion, optimizer_ft, scheduler, batch_size, dataset_dir='.', download_dataset=False, num_epochs=25, num_workers=2):
    self.model = model
    self.device = device
    self.criterion = criterion
    self.optimizer = optimizer_ft
    self.scheduler = scheduler
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.datasets = DataLoader.get_datasets(input_size, dataset_dir=dataset_dir, download=download_dataset)
    self.dataloaders = DataLoader.get_dataloaders(self.datasets, batch_size, num_workers=num_workers)

  def train_model(self, phase):
    since = time.time()

    best_model_wts = copy.deepcopy(self.model.state_dict())
    best_acc = 0.0

    iterations_results = {'Iteration': [], 'Epoch':[], 'Loss': [], 'Accuarcy': []}
    epochs_results = {'Epoch': [], 'Loss': [], 'Accuarcy': []}

    for epoch in range(self.num_epochs):
        print(f'Epoch {epoch}/{self.num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train()  # Set model to training mode
            else:
                self.model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for iter, (inputs, labels) in enumerate(self.dataloaders[phase]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    ###############
                    if torch.sum(torch.isnan(outputs)):
                      import pdb
                      pdb.set_trace()

                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    ###############
                    if torch.sum(torch.isnan(loss)):
                      import pdb
                      pdb.set_trace()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # statistics
                iter_loss = loss.item() * inputs.size(0)
                running_loss += iter_loss
                iter_corrects = torch.sum(preds == labels.data)
                running_corrects += iter_corrects
                ## collect data
                iterations_results['Iteration'].append(iter)
                iterations_results['Epoch'].append(epoch)
                iterations_results['Loss'].append(iter_loss)
                iterations_results['Accuarcy'].append(iter_corrects.cpu().numpy()[0])

            if phase == 'train':
                self.scheduler.step()

            epoch_loss = running_loss / len(self.datasets[phase])
            epoch_acc = running_corrects.double() / len(self.datasets[phase])
            ## collect data
            epochs_results['Epoch'].append(epoch)
            epochs_results['Loss'].append(epoch_loss)
            epochs_results['Accuarcy'].append(epoch_acc.cpu().numpy()[0])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

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
    model_df = pd.DataFrame({'name': ['criterion', 'optimizer', 'scheduler.step_size', 'scheduler.gamma', 'batch_size'],
                             'value': [str(self.criterion), str(self.optimizer), str(self.scheduler.step_size), str(self.scheduler.gamma), str(self.batch_size)]})
    with pd.ExcelWriter(f'{since}_results.xlsx') as writer: 
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
      test_error = torch.sum(preds != y_test.to(self.device)) / len(y_test)
      print(f'test misclassification error: {test_error.item()}')
      return

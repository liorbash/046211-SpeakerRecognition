import optuna
import torch
from torch import optim, nn
from torchvision.models.feature_extraction import create_feature_extractor

from data_utils import DataLoader
from trainer.Trainer import initialize_resnet_model


def get_model(num_classes):
    model, input_size = initialize_resnet_model(num_classes, feature_extract=False, use_pretrained=True)
    return model


def objective(trial):
    # Generate the model.
    num_classes = 200
    model = get_model(num_classes)  # Already on device

    # Generate the optimizers.
    # log=True, will use log scale to interplolate between lr
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lambda_c = trial.suggest_float("lambda_c", 1e-1, 2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adagrad", "SGD"])

    criterion_nn = nn.CrossEntropyLoss()
    criterion_ccl = ContrastiveCenterLoss(dim_hidden=512, num_classes=num_classes, device=device, lambda_c=lambda_c)

    optimizer_ccl = getattr(optim, optimizer_name)(criterion_ccl.parameters(), lr=lr)

    # Optimizers & Scheduler
    optimizer_nn = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_nn, step_size=7, gamma=0.1)

    # Using average pooling layer output as input to contrastive center loss
    avgpool = create_feature_extractor(model, return_nodes={"avgpool": "avgpool"})

    # Get Data loader
    transforms = torchvision.transforms.Resize((224, 224))
    train_dataset = DataLoader.VoxCaleb1MelSpecDataset('./drive/MyDrive/VoxCeleb1', 'train', transform=transforms,
                                                       download=False, min_max_scale=False, normalize_mel=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
    valid_dataset = DataLoader.VoxCaleb1MelSpecDataset('./drive/MyDrive/VoxCeleb1', 'val', transform=transforms,
                                                       download=False, min_max_scale=False, normalize_mel=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=2)
    # Training of the model.
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            data, target = data.to(device), target.to(device)
            avgpool_outputs = avgpool(data)['avgpool']

            optimizer_nn.zero_grad()
            optimizer_ccl.zero_grad()
            output = model(data)
            loss = criterion_nn(output, target) + criterion_ccl(target, avgpool_outputs)
            loss.backward()
            optimizer_nn.step()
            optimizer_ccl.step()

        # Validation of the model.
        model.eval()
        criterion_ccl.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * batch_size >= n_valid_examples:
                    break
                data, target = data.to(device), target.to(device)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), n_valid_examples)

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(accuracy, epoch)
        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="resnet-fc", direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=10)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned trials: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")

    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

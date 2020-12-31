from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
import os

from utils import BasketballDataset, VideoFilePathToTensor
from C3D import C3D

import copy

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            i = 1

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample["video"]
                labels = sample["action"]
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, torch.max(labels, 1)[1])

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        i += 1
                        print(i*12/27201)


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    data_dir = "./dataset/examples/"

    # 10 - Number of classes of basketball actionsn
    num_classes = 10

    # Batch size for training (change depending on how much memory you have)
    batch_size = 12

    # Number of epochs to train for
    num_epochs = 20

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Initialize C3D Model
    model = C3D(num_classes=101, pretrained=True)
    print(model)
    # change final fully-connected layer to output 10 classes and input to 184320
    set_parameter_requires_grad(model, feature_extract)

    # input of the next hidden layer
    num_ftrs = model.fc7.in_features

    # New Model is trained with 128x176 images
    # Calculation:
    model.fc6 = nn.Linear(15360, num_ftrs, bias=True)
    # input of the final hidden layer
    num_ftrs = model.fc8.in_features
    model.fc8 = nn.Linear(num_ftrs, num_classes, bias=True)
    print(model)

    # Load Dataset
    basketball_dataset = BasketballDataset(annotation_dict="dataset/annotation_dict.json",
                                           label_dict="dataset/labels_dict.json",
                                           transform=transforms.Compose(
                                               [VideoFilePathToTensor(max_len=16, fps=10, padding_mode='last')]))

    train_subset, val_subset = random_split(
    basketball_dataset, [27085, 10000], generator=torch.Generator().manual_seed(1))
    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset=train_subset, shuffle=False, batch_size=batch_size)

    print(type(train_loader))
    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)


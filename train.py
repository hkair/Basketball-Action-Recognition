from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt

from dataset import BasketballDataset

import copy
from tqdm import tqdm
from vidaug import augmentors as vidaug

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            train_n_total = 1

            pbar = tqdm(dataloaders[phase])
            # Iterate over data.
            for sample in pbar:
                inputs = sample["video"]
                labels = sample["action"]
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, torch.max(labels, 1)[1])

                    _, preds = torch.max(outputs, 1)
                    #print(preds)
                    #print(torch.max(labels, 1)[1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

                pbar.set_description('Phase: {} || Epoch: {} || Loss {:.5f} '.format(phase, epoch, running_loss / train_n_total))
                train_n_total += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print(phase, ' training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, val_loss_history, train_loss_history

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        i = 12
        for sample in loader:
            x = sample["video"].to(device=device)
            y = sample["action"].to(device=device)

            scores = model(x)
            print(scores)
            predictions = scores.argmax (1)
            y = y.argmax (1)

            print(y)
            print(predictions)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            print(i/5000)
            i += 12

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

if __name__ == "__main__":
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print("Current Device: ", torch.cuda.current_device())
    print("Device: ", torch.cuda.device(0))
    print("Cuda Is Available: ", torch.cuda.is_available())
    print("Device Count: ", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 10 - Number of classes of basketball actions
    num_classes = 10
    # Batch size for training (change depending on how much memory you have)
    batch_size = 12
    # Number of epochs to train for
    num_epochs = 20
    # Unfreeze Layers
    layers = ['layer3', 'layer4', 'fc']

    # Initialize R(2+1)D Model
    model = models.video.r2plus1d_18(pretrained=False, progress=True)

    # change final fully-connected layer to output 10 classes
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        for layer in layers:
            if layer in name:
                param.requires_grad = True

    # input of the next hidden layer
    num_ftrs = model.fc.in_features
    print(num_ftrs)
    # New Model is trained with 128x176 images
    # Calculation:
    model.fc = nn.Linear(num_ftrs, num_classes, bias=True)
    print(model)

    # Put model into device after updating parameters
    model = model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        print(" ")

    # Transforms
    sometimes = lambda aug: vidaug.Sometimes(0.5, aug)  # Used to apply augmentor with 50% probability
    video_augmentation = vidaug.Sequential([
        sometimes(vidaug.Salt()),
        sometimes(vidaug.Pepper()),
    ], random_order=True)

    #Load Dataset
    basketball_dataset = BasketballDataset(annotation_dict="dataset/annotation_dict.json",
                                           augmented_dict="dataset/augmented_annotation_dict.json")

    train_subset, test_subset = random_split(
    basketball_dataset, [44911, 4990], generator=torch.Generator().manual_seed(1))

    train_subset, val_subset = random_split(
        train_subset, [34931, 9980], generator=torch.Generator().manual_seed(1))

    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_subset, shuffle=False, batch_size=batch_size)

    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    # Train
    optimizer_ft = optim.Adam(params_to_update, lr=0.01)

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model, val_acc_history, train_acc_history, val_loss_history, train_loss_history = train_model(model,
                                                                                                  dataloaders_dict,
                                                                                                  criterion,
                                                                                                  optimizer_ft,
                                                                                                  num_epochs=num_epochs)

    print("Best Validation Loss: ", min(val_loss_history), "Epoch: ", val_loss_history.index(min(val_loss_history)))
    print("Best Training Loss: ", min(train_loss_history), "Epoch: ", train_loss_history.index(min(train_loss_history)))

    # Plot Accuracy
    plt.plot(train_acc_history)
    plt.plot(val_acc_history)

    # Plot Loss
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)

    # Save Model
    PATH = "model/small_dataset/"
    torch.save(model.state_dict(), PATH + "c3d-basketball-overfit.pth")

    # Save Model
    PATH = "model/"
    torch.save(model.state_dict(), PATH + "c3d-basketball.pth")

    # Check Accuracy with Test Set
    check_accuracy(test_loader, model)



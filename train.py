from __future__ import print_function
from __future__ import division

import numpy as np
import copy
import time
from tqdm import tqdm
from easydict import EasyDict
from vidaug import augmentors as vidaug
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, random_split

from dataset import BasketballDataset
from utils.checkpoints import init_session_history, save_weights, load_weights, write_history, read_history, plot_curves
from utils.metrics import get_acc_f1_precision_recall

args = EasyDict({

    'base_model_name': 'r2plus1d_multiclass',
    'pretrained': True,

    # training/model params
    'lr': 0.0001,
    'start_epoch': 1,
    'num_epochs': 25,
    'layers_list': ['layer3', 'layer4', 'fc'],
    'continue_epoch': False,

    # Dataset params
    'num_classes': 10,
    'batch_size': 8,
    'n_total': 49901,
    'test_n': 4990,
    'val_n': 9980,

    # Path params
    'annotation_path': "dataset/annotation_dict.json",
    'augmented_annotation_path': "dataset/augmented_annotation_dict.json",
    'model_path': "model_checkpoints/r2plus1d_augmented-2/",
    'history_path': "histories/history_r2plus1d_augmented-2.txt"
})

def train_model(model, dataloaders, criterion, optimizer, args, start_epoch=1, num_epochs=25):
    """
    Trains the 3D CNN Model
    :param model: Model object that we will train
    :param base_model_name: The base name of the model
    :param dataloaders: A dictionary of train and validation dataloader
    :param criterion: Pytorch Criterion Instance
    :param optimizer: Pytorch Optimizer Instance
    :param num_epochs: Number of epochs during training
    :return: model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_score, val_f1_score, plot_epoch
    """

    # Initializes Session History in the history file
    init_session_history(args)
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    train_f1_score = []
    val_f1_score = []
    plot_epoch = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                train_pred_classes = []
                train_ground_truths = []
            else:
                model.eval()  # Set model to evaluate mode
                val_pred_classes = []
                val_ground_truths = []

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

                    if phase == 'train':
                        train_pred_classes.extend(preds.detach().cpu().numpy())
                        train_ground_truths.extend(torch.max(labels, 1)[1].detach().cpu().numpy())
                    else:
                        val_pred_classes.extend(preds.detach().cpu().numpy())
                        val_ground_truths.extend(torch.max(labels, 1)[1].detach().cpu().numpy())

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

            # Calculate elapsed time
            time_elapsed = time.time() - since
            print(phase, ' training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # For Checkpointing and Confusion Matrix
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_pred_classes = np.asarray(val_pred_classes)
                val_ground_truths = np.asarray(val_ground_truths)
                val_accuracy, val_f1, val_precision, val_recall = get_acc_f1_precision_recall(
                    val_pred_classes, val_ground_truths
                )
                val_f1_score.append(val_f1)
                val_confusion_matrix = np.array_str(confusion_matrix(val_ground_truths, val_pred_classes, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
                print('Epoch: {} || Val_Acc: {} || Val_Loss: {}'.format(
                    epoch, val_accuracy, epoch_loss
                ))
                print(f'val: \n{val_confusion_matrix}')

                # Deep Copy Model if best accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                # set current loss to val loss for write history
                val_loss = epoch_loss

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                train_pred_classes = np.asarray(train_pred_classes)
                train_ground_truths = np.asarray(train_ground_truths)
                train_accuracy, train_f1, train_precision, train_recall = get_acc_f1_precision_recall(
                    train_pred_classes, train_ground_truths
                )
                train_f1_score.append(train_f1)
                train_confusion_matrix = np.array_str(confusion_matrix(train_ground_truths, train_pred_classes, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
                print('Epoch: {} || Train_Acc: {} || Train_Loss: {}'.format(
                    epoch, train_accuracy, epoch_loss
                ))
                print(f'train: \n{train_confusion_matrix}')
                plot_epoch.append(epoch)

                # set current loss to train loss for write history
                train_loss = epoch_loss

        # Save Weights
        model_name = save_weights(model, args, epoch, optimizer)

        # Write History after train and validation phase
        write_history(
            args.history_path,
            model_name,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            train_f1,
            val_f1,
            train_precision,
            val_precision,
            train_recall,
            val_recall,
            train_confusion_matrix,
            val_confusion_matrix
        )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_score, val_f1_score, plot_epoch

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        i = args.batch_size

        pbar = tqdm(loader)
        for sample in pbar:
            x = sample["video"].to(device=device)
            y = sample["action"].to(device=device)

            scores = model(x)
            print(scores)
            predictions = scores.argmax (1)
            y = y.argmax (1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            pbar.set_description('Progress: {}'.format(i/args.test_n))
            i += args.batch_size

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

    # Initialize R(2+1)D Model
    model = models.video.r2plus1d_18(pretrained=args.pretrained, progress=True)

    # change final fully-connected layer to output 10 classes
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        for layer in args.layers_list:
            if layer in name:
                param.requires_grad = True

    # input of the next hidden layer
    num_ftrs = model.fc.in_features
    # New Model is trained with 128x176 images
    # Calculation:
    model.fc = nn.Linear(num_ftrs, args.num_classes, bias=True)
    print(model)

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
    basketball_dataset = BasketballDataset(annotation_dict=args.annotation_path,
                                           augmented_dict=args.augmented_annotation_path)

    train_subset, test_subset = random_split(
    basketball_dataset, [args.n_total-args.test_n, args.test_n], generator=torch.Generator().manual_seed(1))

    train_subset, val_subset = random_split(
        train_subset, [args.n_total-args.test_n-args.val_n, args.val_n], generator=torch.Generator().manual_seed(1))

    train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_subset, shuffle=False, batch_size=args.batch_size)

    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    # Train
    optimizer_ft = optim.Adam(params_to_update, lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    if args.continue_epoch:
        model = load_weights(model, args)

    if torch.cuda.is_available():
        # Put model into device after updating parameters
        model = model.to(device)
        criterion = criterion.to(device)

    # Train and evaluate
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_score, val_f1_score, plot_epoch = train_model(model,
                                                                                                                                            dataloaders_dict,
                                                                                                                                            criterion,
                                                                                                                                            optimizer_ft,
                                                                                                                                            args,
                                                                                                                                            start_epoch=args.start_epoch,
                                                                                                                                            num_epochs=args.num_epochs)

    print("Best Validation Loss: ", min(val_loss_history), "Epoch: ", val_loss_history.index(min(val_loss_history)))
    print("Best Training Loss: ", min(train_loss_history), "Epoch: ", train_loss_history.index(min(train_loss_history)))

    # Plot Final Curve
    plot_curves(
        args.base_model_name,
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        train_f1_score,
        val_f1_score,
        plot_epoch
    )

    # Read History
    read_history(args.history_path)

    # Check Accuracy with Test Set
    check_accuracy(test_loader, model)



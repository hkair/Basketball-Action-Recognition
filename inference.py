from __future__ import print_function
from __future__ import division

import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, random_split

from dataset import BasketballDataset
from utils.checkpoints import load_weights
from utils.metrics import get_acc_f1_precision_recall

args = EasyDict({

    'base_model_name': 'r2plus1d_multiclass',
    'pretrained': True,

    # training/model params
    'lr': 0.0001,
    'start_epoch': 19,
    # 19, 15, 3

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

def batch_to_framelist(batch):
    # (batch, c, t, h, w) -> (batch, t, h, w, c)
    batch = batch.permute(0, 2, 3 ,4 ,1)
    framelist = batch.cpu().numpy()
    return framelist

def inference(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    pred_classes = []
    ground_truths = []
    f1_score = []

    correct = []
    incorrect = []

    correct_softmax, incorrect_softmax = [], []

    with torch.no_grad():
        i = args.batch_size

        pbar = tqdm(loader)
        for sample in pbar:
            raw_data = batch_to_framelist(sample["video"])
            x = sample["video"].to(device=device)
            y = sample["action"].to(device=device)

            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            y_ = y.argmax (1)

            num_correct += (preds == y_).sum()
            num_samples += preds.size(0)

            # predicted class
            pred_class = preds.detach().cpu().numpy()
            pred_classes.extend(pred_class)
            # ground truth
            gt = torch.max(y, 1)[1].detach().cpu().numpy()
            ground_truths.extend(gt)

            # Convert back into cpu
            softmax_predictions = torch.softmax(outputs, dim=-1).cpu().numpy().tolist()
            softmax_preds = [max(pred) for pred in softmax_predictions]

            for framedata, softpred, softpreds, predlabel, label in zip(raw_data, softmax_preds, softmax_predictions, pred_class, gt):
                # If correct
                if predlabel == label:
                    correct.append({
                        'frames': framedata,
                        'softpred': softpred,
                        'softpreds': softpreds,
                        'prediction': predlabel,
                        'label': label,
                    })
                    correct_softmax.append(softpred)

                # If incorrect
                else:
                    incorrect.append({
                        'frames': framedata,
                        'softpred': softpred,
                        'softpreds': softpreds,
                        'prediction': predlabel,
                        'label': label,
                    })
                    correct_softmax.append(softpred)

            pbar.set_description('Progress: {}'.format(i/args.test_n))
            i += args.batch_size

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

    pred_classes = np.asarray(pred_classes)
    ground_truths = np.asarray(ground_truths)
    val_accuracy, val_f1, val_precision, val_recall = get_acc_f1_precision_recall(
        pred_classes, ground_truths
    )
    f1_score.append(val_f1)

    confusion_matrix_ = np.array_str(
        confusion_matrix(ground_truths, pred_classes, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(f'val: \n{confusion_matrix_}')

    predictions = {
        "correct": correct,
        "incorrect": incorrect,
    }
    return predictions, confusion_matrix_

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

    # input of the next hidden layer
    num_ftrs = model.fc.in_features
    # New Model is trained with 128x176 images
    # Calculation:
    model.fc = nn.Linear(num_ftrs, args.num_classes, bias=True)
    print(model)

    model = load_weights(model, args)

    if torch.cuda.is_available():
        # Put model into device after updating parameters
        model = model.to(device)

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

    # Check Accuracy with Test Set
    predictions, confusion_matrix_ = inference(test_loader, model)
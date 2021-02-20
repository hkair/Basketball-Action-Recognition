import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def get_acc_f1_precision_recall(pred_classes, ground_truths, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    Given two linear arrays of predicted classes and ground truths, return accuracy, f1 score, precision and recall
    :param pred_classes: classes predicted by model
    :param ground_truths: ground truths for predictions
    :return: tuple of accuracy, f1, precision, recall
    """

    print(pred_classes)
    print(ground_truths)

    accuracy = np.mean((pred_classes == ground_truths)).astype(np.float)
    f1 = f1_score(ground_truths, pred_classes, labels=labels, average='micro')
    precision = precision_score(ground_truths, pred_classes, labels=labels, average='micro')
    recall = recall_score(ground_truths, pred_classes, labels=labels, average='micro')

    return accuracy, f1, precision, recall
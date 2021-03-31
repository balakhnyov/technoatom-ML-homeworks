import numpy as np

file = np.loadtxt('HW2_labels.txt',  delimiter=',')
y_predict, y_true = file[:, :2], file[:, -1]
print(y_predict)
print(np.unique(y_true))


def accuracy_score(y_true, y_predict, percent=None):
    if percent is None:
        p = 0.5
    else:
        p = percent / 100
    TP = np.sum((y_predict[:, 1][y_true == 1] >= p).astype('float64'))
    TN = np.sum((y_predict[:, 0][y_true == 0] >= p).astype('float64'))
    FP = np.sum((y_predict[:, 1][y_true == 0] >= p).astype('float64'))
    FN = np.sum((y_predict[:, 0][y_true == 1] >= p).astype('float64'))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


def precision_score(y_true, y_predict, percent=None):
    if percent is None:
        p = 0.5
    else:
        p = percent / 100

    TP = np.sum((y_predict[:, 1][y_true == 1] >= p).astype('float64'))
    FP = np.sum((y_predict[:, 1][y_true == 0] >= p).astype('float64'))
    precision = TP / (TP + FP)
    return precision


def recall_score(y_true, y_predict, percent=None):
    if percent is None:
        p = 0.5
    else:
        p = percent / 100

    TP = np.sum((y_predict[:, 1][y_true == 1] >= p).astype('float64'))
    FN = np.sum((y_predict[:, 0][y_true == 1] >= p).astype('float64'))
    recall = TP / (TP + FN)
    return recall


def lift_score(y_true, y_predict, percent=None):
    if percent is None:
        p = 0.5
    else:
        p = percent / 100

    TP = np.sum((y_predict[:, 1][y_true == 1] >= p).astype('float64'))
    TN = np.sum((y_predict[:, 0][y_true == 0] >= p).astype('float64'))
    FP = np.sum((y_predict[:, 1][y_true == 0] >= p).astype('float64'))
    FN = np.sum((y_predict[:, 0][y_true == 1] >= p).astype('float64'))
    lift = (TP / (TP + FP)) / ((TP + FN) / (TP + TN + FP + FN))
    return lift


def f1_score(y_true, y_predict, percent=None):
    precision = precision_score(y_true, y_predict, percent)
    recall = recall_score(y_true, y_predict, percent)
    return 2 * (precision * recall) / (precision + recall)
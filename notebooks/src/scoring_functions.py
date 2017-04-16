
import numpy as np


def my_brier_scorer(predictor, X, y):
    probs = predictor.predict_proba(X)
    sorted_probs = []
    score = 0
    for i in range(len(probs[0])):
        list_aux = []
        for j in range(25):
            if probs[j][i].shape[0] > 1:
                list_aux.append(probs[j][i][1])
            else:
                list_aux.append(0)
        sorted_probs.append(list_aux)
    for i in range(y.shape[0]):
        aux = np.square(sorted_probs[i] - y[i])
        score += np.mean(aux[0:4]) + np.mean(aux[4:7]) + \
            np.mean(aux[7:16]) + np.mean(aux[16:25])
    return -score / y.shape[0]


def my_accuracy_scorer(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 0:4] == ypred[i, 0:4]).all():
            score += 0.25
        if (y[i, 4:7] == ypred[i, 4:7]).all():
            score += 0.25
        score += (8 - abs(np.argmax(y[i, 7:16] == 1) -
                          np.argmax(ypred[i, 7:16] == 1))) / 8 * 0.25
        score += (8 - abs(np.argmax(y[i, 16:25] == 1) -
                          np.argmax(ypred[i, 16:25] == 1))) / 8 * 0.25
    return score / y.shape[0]

def my_accuracy_scorer2(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 0:7] == ypred[i, 0:7]).all():
            score += 0.5
        score += (8 - abs(np.argmax(y[i, 7:16] == 1) -
                          np.argmax(ypred[i, 7:16] == 1)))**2 / 64 * 0.25
        score += (8 - abs(np.argmax(y[i, 16:25] == 1) -
                          np.argmax(ypred[i, 16:25] == 1)))**2 / 64 * 0.25
    return score / y.shape[0]



import numpy as np


def my_brier_scorer(predictor, X, y):
    probs = predictor.predict_proba(X)
    sorted_probs = []
    score = 0
    for i in range(len(probs[0])):
        list_aux = []
        for j in range(26):
            if probs[j][i].shape[0] > 1:
                list_aux.append(probs[j][i][1])
            else:
                list_aux.append(0)
        sorted_probs.append(list_aux)
    for i in range(y.shape[0]):
        aux = np.square(sorted_probs[i] - y[i])
        score += np.mean(aux[0:5]) + np.mean(aux[5:8]) + \
            np.mean(aux[8:17]) + np.mean(aux[17:26])
    return score / y.shape[0]


def my_accuracy_scorer(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 0:5] == ypred[i, 0:5]).all():
            score += 0.25
        if (y[i, 5:8] == ypred[i, 5:8]).all():
            score += 0.25
        score += (8 - abs(np.argmax(y[i, 8:17] == 1) -
                          np.argmax(ypred[i, 8:17] == 1))) / 8 * 0.25
        score += (8 - abs(np.argmax(y[i, 17:26] == 1) -
                          np.argmax(ypred[i, 17:26] == 1))) / 8 * 0.25
    return score / y.shape[0]

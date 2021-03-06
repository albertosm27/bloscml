
import numpy as np
import random
import pandas as pd


def brier(predictor, X, y):
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
        score += np.sum(aux[0:4]) + np.sum(aux[4:7]) + \
            np.sum(aux[7:16]) + np.sum(aux[16:25])
    return -score / y.shape[0]


def balanced(predictor, X, y):
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


def codec_filter(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 0:7] == ypred[i, 0:7]).all():
            score += 1
    return score / y.shape[0]


def codec(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 0:4] == ypred[i, 0:4]).all():
            score += 1
    return score / y.shape[0]


def filter_(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 4:7] == ypred[i, 4:7]).all():
            score += 1
    return score / y.shape[0]


def c_level(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 7:16] == ypred[i, 7:16]).all():
            score += 1
    return score / y.shape[0]


def c_level_nice(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        score += (8 - abs(np.argmax(y[i, 7:16] == 1) -
                          np.argmax(ypred[i, 7:16] == 1)))**2 / 64
    return score / y.shape[0]


def block(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 16:25] == ypred[i, 16:25]).all():
            score += 1
    return score / y.shape[0]


def block_nice(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        score += (8 - abs(np.argmax(y[i, 16:25] == 1) -
                          np.argmax(ypred[i, 16:25] == 1)))**2 / 64
    return score / y.shape[0]


def cl_block(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i, 7:25] == ypred[i, 7:25]).all():
            score += 1
    return score / y.shape[0]


def cl_block_nice(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        score += (8 - abs(np.argmax(y[i, 7:16] == 1) -
                          np.argmax(ypred[i, 7:16] == 1)))**2 / 64 * 0.5
        score += (8 - abs(np.argmax(y[i, 16:25] == 1) -
                          np.argmax(ypred[i, 16:25] == 1)))**2 / 64 * 0.5
    return score / y.shape[0]

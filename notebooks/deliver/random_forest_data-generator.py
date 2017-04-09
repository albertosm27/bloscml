
import time
import matplotlib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve

START_TIME = time.time()
pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})

IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int',
              'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3',
              'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd']
OUT_FILTER = ['Noshuffle', 'Shuffle', 'Bitshuffle']
OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_BLOCKS = ['Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128',
              'Block_256', 'Block_512', 'Block_1024', 'Block_2048']
OUT_OPTIONS = OUT_CODEC + OUT_FILTER + OUT_LEVELS + OUT_BLOCKS


def my_brier_scorer(predictor, X, y):
    probs = predictor.predict_proba(X)
    sorted_probs = []
    score = 0
    for i in range(len(probs[0])):
        sorted_probs.append([probs[j][i][1] for j in range(26)])
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
        score += abs(y[i, 8:17].index(1) - ypred[i, 8:17].index(1)) / 8 * 0.25
        score += abs(y[i, 17:26].index(1) -
                     ypred[i, 17:26].index(1)) / 8 * 0.25
    return score / y.shape[0]


NUM_TRIALS = 30
SCORES = [None, my_brier_scorer, my_accuracy_scorer]

DF = pd.read_csv('../data/training_data.csv', sep='\t')
X, Y = DF[IN_OPTIONS].values, DF[OUT_OPTIONS].values

P_GRID = {'criterion': ['gini', 'entropy'],
          'bootstrap': [True, False],
          'class_weight': [None, 'balanced']}

RFC = RandomForestClassifier(n_estimators=30, n_jobs=-1)

NON_NESTED_SCORES = []
NESTED_SCORES = []
ESTIMATOR_V_CURVES = []
FEATURES_V_CURVES = []
DEPTH_V_CURVES = []
ESTIMATORS = []

for score in SCORES:
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)
    estimators_valid_curves = []
    features_valid_curves = []
    depth_valid_curves = []
    estimators = []
    print('Starting with ' + str(score))
    for i in range(NUM_TRIALS):
        print(str(score) + ' trial number ' + str(i))
        inner_cv = ShuffleSplit(n_splits=10, test_size=0.25)
        outer_cv = ShuffleSplit(n_splits=10, test_size=0.25)

        clf = GridSearchCV(estimator=RFC, param_grid=P_GRID,
                           cv=inner_cv, n_jobs=-1, scoring=score)
        clf.fit(X, Y)
        non_nested_scores[i] = clf.best_score_
        print('Non nested scores completed')

        nested_score = cross_val_score(
            clf, X=X, y=Y, cv=outer_cv, scoring=score, n_jobs=1)
        nested_scores[i] = nested_score.mean()
        print('Nested scores completed')

        estimators.append(clf.best_estimator_)
        estimators_valid_curves.append(
            validation_curve(clf.best_estimator_, X, Y,
                             param_name="n_estimators",
                             param_range=[30, 40, 50, 60, 70, 80, 90, 100],
                             cv=outer_cv, scoring=score, n_jobs=-1))
        print('Estimators validations curves completed')
        features_valid_curves.append(
            validation_curve(clf.best_estimator_,
                             X, Y, param_name="max_features",
                             param_range=[12, 14, 16, 18, 20, 22, 24],
                             cv=outer_cv, scoring=score, n_jobs=-1))
        print('Features validations curves completed')
        depth_valid_curves.append(
            validation_curve(clf.best_estimator_, X, Y, param_name="max_depth",
                             param_range=[6, 8, 10, 12, 14, 16, 18, 20],
                             cv=outer_cv, scoring=score, n_jobs=-1))
        print('Depth validations curves completed')
    NON_NESTED_SCORES.append(non_nested_scores)
    NESTED_SCORES.append(nested_scores)
    ESTIMATOR_V_CURVES.append(estimators_valid_curves)
    FEATURES_V_CURVES.append(features_valid_curves)
    DEPTH_V_CURVES.append(depth_valid_curves)
    ESTIMATORS.append(estimators)
joblib.dump(NON_NESTED_SCORES, 'Non_nested_scores.pkl')
joblib.dump(NESTED_SCORES, 'Nested_scores.pkl')
joblib.dump(ESTIMATOR_V_CURVES, 'Estim_v_curve.pkl')
joblib.dump(FEATURES_V_CURVES, 'Feat_v_curve.pkl')
joblib.dump(DEPTH_V_CURVES, 'Depth_v_curve.pkl')
joblib.dump(ESTIMATORS, 'Best_estimators.pkl')
print("--- %s minutes ---" % str((time.time() - START_TIME) / 60))

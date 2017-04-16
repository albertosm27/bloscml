
import sys
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import scale
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.externals import joblib
from scoring_functions import my_accuracy_scorer
from scoring_functions import my_brier_scorer


NUM_TRIALS = 20
SCORES = [my_accuracy_scorer, my_brier_scorer]
DF = pd.read_csv('../data/training_data.csv', sep='\t')
IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int',
              'is_Float', 'is_String', 'Type_Size', 'Chunk_Size', 'Mean',
              'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3',
              'N_Streaks', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed',
              'LZ4_CRate', 'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Zstd']
OUT_FILTER = ['Noshuffle', 'Shuffle', 'Bitshuffle']
OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_BLOCKS = ['Block_8', 'Block_16', 'Block_32', 'Block_64',
              'Block_128', 'Block_256', 'Block_512', 'Block_1024',
              'Block_2048']
OUT_OPTIONS = OUT_CODEC + OUT_FILTER + OUT_LEVELS + OUT_BLOCKS
X, Y = scale(DF[IN_OPTIONS].values), DF[OUT_OPTIONS].values
ESTIMATORS = []
ESTIMATORS.append(('KNei', KNeighborsClassifier(n_jobs=-1),
                   {'n_neighbors': [5, 10, 15, 30, 50],
                    'weights': ['uniform', 'distance'],
                    }, 2))
ESTIMATORS.append(('SVC', MultiOutputClassifier(SVC(
    decision_function_shape='ovr')),
    {'estimator__C': [1, 10, 100, 1000],
     'estimator__gamma': [0.1, 0.01, 0.001]}, 1))
ESTIMATORS.append(('RFC', RandomForestClassifier(
    n_estimators=40, max_depth=14, n_jobs=-1),
    {'criterion': ['gini', 'entropy'],
     'bootstrap': [True, False],
     'class_weight': [None, 'balanced']}, 2))
ESTIMATORS.append(('ETC', ExtraTreesClassifier(
    n_estimators=40, max_depth=14, n_jobs=-1),
    {'criterion': ['gini', 'entropy'],
     'bootstrap': [True, False],
     'class_weight': [None, 'balanced']}, 2))
TOTAL = len(ESTIMATORS) * len(SCORES) * NUM_TRIALS - NUM_TRIALS


def main():
    count = 0
    for estimator_opt, estimator, p_grid, idx in ESTIMATORS:
        if estimator_opt == 'SVC':
            splits = 4
        else:
            splits = 10
        for score in SCORES[:idx]:
            non_nested_scores = np.zeros(NUM_TRIALS)
            non_nested_estimators = []
            nested_scores = np.zeros(NUM_TRIALS)
            nested_estimators = []
            for i in range(NUM_TRIALS):
                start_time = time.time()
                print('Estimator %s, Scorer: %s iteration %d ---- %.2f %%' %
                      (estimator_opt, score.__name__, i, count / TOTAL * 100))
                inner_cv = ShuffleSplit(n_splits=splits, test_size=0.25)
                outer_cv = ShuffleSplit(n_splits=splits, test_size=0.25)
                clf = GridSearchCV(estimator=estimator, param_grid=p_grid,
                                   cv=inner_cv, n_jobs=-1, scoring=score)
                clf.fit(X, Y)
                non_nested_scores[i] = clf.best_score_
                non_nested_estimators.append(clf.best_estimator_)

                winner_estimators = []
                outer_scores = []
                for train_index, test_index in outer_cv.split(X, Y):
                    clf = GridSearchCV(estimator=estimator, param_grid=p_grid,
                                       cv=inner_cv, n_jobs=-1, scoring=score)
                    clf.fit(X[train_index], Y[train_index])
                    winner_estimators.append(clf.best_estimator_)
                    outer_scores.append(
                        score(clf, X[test_index], Y[test_index]))
                nested_scores[i] = np.mean(outer_scores)
                nested_estimators.append(winner_estimators)
                count += 1
                print('Time: %f minutes' % ((time.time() - start_time) / 60))
            joblib.dump(non_nested_scores, estimator_opt +
                        'non_nested_scores_' +
                        score.__name__ + '.pkl', compress=('zlib', 3))
            joblib.dump(nested_scores, estimator_opt + 'nested_scores_' +
                        score.__name__ + '.pkl', compress=('zlib', 3))
            joblib.dump(non_nested_estimators, estimator_opt +
                        'non_nested_estimators_' + score.__name__ + '.pkl',
                        compress=('zlib', 3))
            joblib.dump(nested_estimators, estimator_opt +
                        'nested_estimators_' +
                        score.__name__ + '.pkl', compress=('zlib', 3))


if __name__ == "__main__":
    main()


import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.externals import joblib
from scoring_functions import my_accuracy_scorer
from scoring_functions import my_brier_scorer


NUM_TRIALS = 30
SCORES = [my_accuracy_scorer, my_brier_scorer]
DF = pd.read_csv('../data/training_data.csv', sep='\t')
IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int',
              'is_Float', 'is_String', 'Type_Size', 'Chunk_Size', 'Mean',
              'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3',
              'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd']
OUT_FILTER = ['Noshuffle', 'Shuffle', 'Bitshuffle']
OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_BLOCKS = ['Block_8', 'Block_16', 'Block_32', 'Block_64',
              'Block_128', 'Block_256', 'Block_512', 'Block_1024',
              'Block_2048']
OUT_OPTIONS = OUT_CODEC + OUT_FILTER + OUT_LEVELS + OUT_BLOCKS
X, Y = DF[IN_OPTIONS].values, DF[OUT_OPTIONS].values


def main():
    count = 0
    estimator_opt = sys.argv[1]
    if estimator_opt == 'RFC':
        estimator = RandomForestClassifier(
            n_estimators=40, max_depth=16, n_jobs=-1)
        p_grid = {'criterion': ['gini', 'entropy'],
                  'bootstrap': [True, False],
                  'class_weight': [None, 'balanced']}
    else:
        print('Wrong arg passed %s' % estimator_opt)
        print('Expected values: RFC')
        exit()
    for score in SCORES:
        non_nested_scores = np.zeros(NUM_TRIALS)
        non_nested_estimators = []
        nested_scores = np.zeros(NUM_TRIALS)
        nested_estimators = []
        for i in range(NUM_TRIALS):
            print('Scorer: %s iteration %d ---- %.2f %%' %
                  (score.__name__, i, count / 60 * 100))
            inner_cv = ShuffleSplit(n_splits=10, test_size=0.25)
            outer_cv = ShuffleSplit(n_splits=10, test_size=0.25)
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
                outer_scores.append(score(clf, X[test_index], Y[test_index]))
            nested_scores[i] = np.mean(outer_scores)
            nested_estimators.append(winner_estimators)
        joblib.dump(non_nested_scores, 'non_nested_scores_' +
                    score.__name__ + '.pkl')
        joblib.dump(nested_scores, 'nested_scores_' +
                    score.__name__ + '.pkl')
        joblib.dump(non_nested_estimators, 'non_nested_estimators_' +
                    score.__name__ + '.pkl')
        joblib.dump(nested_estimators, 'nested_estimators_' +
                    score.__name__ + '.pkl')


if __name__ == "__main__":
    main()

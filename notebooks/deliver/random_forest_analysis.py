
# coding: utf-8

# # Alogirtmos Random Forest

# ## Objetivos
# * Analizar las distintas opciones de scikit-learn a la hora de crear algoritmos del tipo Random Forest

# In[1]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas, scikit-learn')


# In[2]:

import os
import sys
sys.path.append("../src/")

from IPython.display import display
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# In[3]:

IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd']
OUT_FILTER = ['Noshuffle', 'Shuffle', 'Bitshuffle'] 
OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_BLOCKS = ['Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128', 'Block_256', 'Block_512', 'Block_1024', 'Block_2048']
OUT_OPTIONS = OUT_CODEC + OUT_FILTER + OUT_LEVELS + OUT_BLOCKS


# In[4]:

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
        score += np.mean(aux[0:5]) + np.mean(aux[5:8]) + np.mean(aux[8:17]) + np.mean(aux[17:26])
    return score/y.shape[0]


# In[5]:

def my_accuracy_scorer(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i,0:5] == ypred[i,0:5]).all():
            score += 0.25
        if (y[i,5:8] == ypred[i,5:8]).all():
            score += 0.25
        score += (8 - abs(np.argmax(y[i,8:17] == 1) - np.argmax(ypred[i,8:17] == 1))) / 8 * 0.25
        score += (8 - abs(np.argmax(y[i,17:26] == 1) - np.argmax(ypred[i,17:26] == 1))) / 8 * 0.25
    return score/y.shape[0]


# In[6]:

NUM_TRIALS = 30
SCORES = [None, my_brier_scorer, my_accuracy_scorer]


# ## Random forest nested cross-validation

# In[7]:

df = pd.read_csv('../data/training_data.csv', sep='\t')
X, Y = df[IN_OPTIONS].values, df[OUT_OPTIONS].values


# In[8]:




# In[9]:

rfc = RandomForestClassifier(n_estimators=50, n_jobs=-1)


# In[10]:

def plot_learning_curve(estimator, title, X, y, scoring, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#FF3B30")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#4CD964")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#FF3B30",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#4CD964",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[11]:

title = "Learning Curves (Random Forest)"
cv = ShuffleSplit(n_splits=20, test_size=0.1)
for score in SCORES:
    plot_learning_curve(rfc, title, X, Y, scoring=score, ylim=(0, 1.01), cv=cv, n_jobs=-1,
                       train_sizes=np.linspace(.1, 1.0, 10))


# In[12]:

def plot_validation_curve(rfc, X, Y, param_name, param_range, cv, scoring):
    train_scores, test_scores = validation_curve(rfc, X, Y, param_name=param_name,
                                                        param_range=param_range, 
                                                        cv=cv, scoring=scoring, n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title("Validation Curve with Random Forest")
    plt.xlabel(param_name)
    plt.ylabel("Score %s" % str(scoring))
    plt.ylim(0.0, 1.1)
    lw = 2
    
    plt.plot(param_range, train_scores_mean, label="Training score",
             color="#FF9500", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="#FF9500", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="#007AFF", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="#007AFF", lw=lw)
    plt.legend(loc="best")
    plt.show()


# In[13]:

PARAM_NAMES = ['n_estimators', 'max_features', 'max_depth']
PARAM_RANGES = [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                [8, 10, 12, 14, 16, 18, 20, 22, 24],
                [6, 8, 10, 12, 14, 16, 18, 20]]
for i in range(3):
    for score in SCORES:
        plot_validation_curve(rfc, X, Y, param_name=PARAM_NAMES[i], param_range=PARAM_RANGES[i],
                              cv=cv, scoring=score)


# In[14]:

all_non_nested_scores = []
all_nested_scores = []
all_estimators = []
p_grid = {'criterion': ['gini', 'entropy'],
          'bootstrap': [True, False],
          'class_weight': [None, 'balanced']}


# In[22]:

get_ipython().run_cell_magic('time', '', "count = 0\nrfc = RandomForestClassifier(n_estimators=40, max_depth=16, n_jobs=-1)\nfor score in SCORES:\n    non_nested_scores = np.zeros(NUM_TRIALS)\n    nested_scores = np.zeros(NUM_TRIALS)\n    estimators_valid_curves = []\n    features_valid_curves = []\n    depth_valid_curves = []\n    estimators = []\n    print('Starting with ' + str(score))\n    for i in range(NUM_TRIALS):\n        print(str(score) + ' trial number ' + str(i))\n        inner_cv = ShuffleSplit(n_splits=10, test_size=0.25)\n        outer_cv = ShuffleSplit(n_splits=10, test_size=0.25)\n        \n        clf = GridSearchCV(estimator=rfc, param_grid=p_grid, cv=inner_cv, n_jobs=-1, scoring=score)\n        clf.fit(X, Y)\n        non_nested_scores[i] = clf.best_score_\n        count += 1\n        print('Non nested scores completed -------------- %.2f %%' % (count / 180))\n        \n        nested_score = cross_val_score(clf, X=X, y=Y, cv=outer_cv, scoring=score, n_jobs=1)\n        nested_scores[i] = nested_score.mean()\n        count += 1\n        print('Nested scores completed ------------------ %.2f %%' % (count / 180))\n        estimators.append(clf.best_estimator_)\n    all_non_nested_scores.append(non_nested_scores)\n    all_nested_scores.append(nested_scores)\n    all_estimators.append(estimators)\njoblib.dump(all_non_nested_scores, 'Non_nested_scores.pkl')\njoblib.dump(all_nested_scores, 'Nested_scores.pkl')\njoblib.dump(all_estimators, 'Estimators.pkl')")


# In[18]:

for non_nested_scores, nested_scores in (all_non_nested_scores, all_nested_scores):
    score_difference = non_nested_scores - nested_scores

    print("Average difference of {0:6f} with std. dev. of {1:6f}."
          .format(score_difference.mean(), score_difference.std()))

    # Plot scores on each trial for nested and non-nested CV
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
               ["Non-Nested CV", "Nested CV"],
               bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
              x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
               ["Non-Nested CV - Nested CV Score"],
               bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")

    plt.show()


# In[ ]:




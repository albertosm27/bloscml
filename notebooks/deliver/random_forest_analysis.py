
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

import pandas as pd
from scoring_functions import my_accuracy_scorer
from scoring_functions import my_brier_scorer

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# In[24]:

SCORES = [my_accuracy_scorer, my_brier_scorer]
IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd']
OUT_FILTER = ['Noshuffle', 'Shuffle', 'Bitshuffle'] 
OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_BLOCKS = ['Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128', 'Block_256', 'Block_512', 'Block_1024', 'Block_2048']
OUT_OPTIONS = OUT_CODEC + OUT_FILTER + OUT_LEVELS + OUT_BLOCKS


# In[4]:

df = pd.read_csv('../data/training_data.csv', sep='\t')
X, Y = df[IN_OPTIONS].values, df[OUT_OPTIONS].values


# In[5]:

rfc = RandomForestClassifier(n_estimators=50, n_jobs=-1)


# ## LEARNING CURVES

# In[6]:

title = "Learning Curves (Random Forest)"
cv = ShuffleSplit(n_splits=20, test_size=0.1)
for score in SCORES:
    plot_learning_curve(rfc, title, X, Y, scoring=score, ylim=(0, 1.01), cv=cv, n_jobs=-1,
                       train_sizes=np.linspace(.1, 1.0, 10))


# ## VALIDATIONS CURVES

# In[8]:

PARAM_NAMES = ['n_estimators', 'max_features', 'max_depth']
PARAM_RANGES = [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                [8, 10, 12, 14, 16, 18, 20, 22, 24],
                [6, 8, 10, 12, 14, 16, 18, 20]]
for i in range(3):
    for score in SCORES:
        plot_validation_curve(rfc, X, Y, param_name=PARAM_NAMES[i], param_range=PARAM_RANGES[i],
                              cv=cv, scoring=score)


# In[9]:




# In[11]:




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




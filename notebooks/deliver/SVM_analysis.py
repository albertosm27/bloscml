
# coding: utf-8

# # Algoritmos SVM

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
import pandas as pd
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from multioutput_chained import ChainedMultiOutputClassifier
from sklearn.preprocessing import scale
import ml_plots as mp
import scoring_functions as sf

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# In[3]:

SCORES = [(sf.balanced, 0.2, 1.01)]
IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar',
              'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1',
              'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Zstd']
OUT_FILTER = ['Noshuffle', 'Shuffle', 'Bitshuffle'] 
OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8','CL9']
OUT_BLOCKS = ['Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128',
              'Block_256', 'Block_512', 'Block_1024', 'Block_2048']
OUT_OPTIONS = OUT_CODEC + OUT_FILTER + OUT_LEVELS + OUT_BLOCKS


# In[4]:

df = pd.read_csv('../data/training_data.csv', sep='\t')
X, Y = scale(df[IN_OPTIONS].values), df[OUT_OPTIONS].values


# In[5]:

clf = LinearSVC()
clf = MultiOutputClassifier(clf)


# ## SVM rbf - Curva de aprendizaje

# ### svc = SVC(kernel='rbf')
# clf = MultiOutputClassifier(svc)
# title = "Learning Curves (rbf SVC)"
# cv = ShuffleSplit(n_splits=10, test_size=0.1)
# fig = plt.figure(figsize=(20,8))
# n = 121
# for score in SCORES:
#     mp.plot_learning_curve(fig.add_subplot(n), clf, title, X, Y, scoring=score[0], ylim=(score[1], score[2]), cv=cv, n_jobs=-1,
#                        train_sizes=np.linspace(.1, 1.0, 10))
#     n += 1

# ## SVM rbf - Validación cruzada de hiperparámetros

# In[6]:

nested_clf = joblib.load(
    '../src/SVCnested_estimators_my_accuracy_scorer.pkl')
non_nested_clf = joblib.load(
    '../src/SVCnon_nested_estimators_my_accuracy_scorer.pkl')
mp.print_nested_winners(nested_clf, non_nested_clf)


# In[7]:

del nested_clf
del non_nested_clf


# ## SVM rbf - Resultados

# In[8]:

clf1 = MultiOutputClassifier(
    SVC(C=1000, gamma=0.1, decision_function_shape='ovr'))
cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=1)
mp.cross_val_report(clf1, cv, X, Y, True)


# In[9]:

CUSTOM3_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'Sd', 'BLZ_CSpeed',
              'Mean', 'Kurt', 'Skew', 'N_Streaks', 'Max', 'Min']
X = scale(df[CUSTOM3_IN].values)
clf1 = MultiOutputClassifier(
    SVC(C=1000, gamma=0.1, decision_function_shape='ovr'))
mp.cross_val_report(clf1, cv, X, Y, True)


# In[10]:

CUSTOM2_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'Sd', 'BLZ_CSpeed',
              'Mean', 'Kurt', 'Skew', 'Max', 'Min']
X = scale(df[CUSTOM2_IN].values)
clf1 = MultiOutputClassifier(
    SVC(C=1000, gamma=0.1, decision_function_shape='ovr'))
mp.cross_val_report(clf1, cv, X, Y, True)


# In[11]:

CUSTOM_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'BLZ_CSpeed']
X = scale(df[CUSTOM_IN].values)
clf1 = MultiOutputClassifier(
    SVC(C=1000, gamma=0.1, decision_function_shape='ovr'))
mp.cross_val_report(clf1, cv, X, Y, True)


# In[12]:

CUSTOM_IN = ['IN_CS', 'IN_DS', 'IN_CR']
X = scale(df[CUSTOM_IN].values)
clf1 = MultiOutputClassifier(
    SVC(C=1000, gamma=0.1, decision_function_shape='ovr'))
mp.cross_val_report(clf1, cv, X, Y, True)


# In[13]:

CUSTOM3_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'Sd', 'BLZ_CSpeed',
              'Mean', 'Kurt', 'Skew', 'N_Streaks', 'Max', 'Min']
X = scale(df[CUSTOM3_IN].values)
clf1 = ChainedMultiOutputClassifier(
    SVC(C=1000, gamma=0.1, decision_function_shape='ovr'))
mp.cross_val_report(clf1, cv, X, Y, True)


# ## LinearSVC - Curva de aprendizaje

# In[14]:

title = "Learning Curves (LinearSVC)"
cv = ShuffleSplit(n_splits=10, test_size=0.1)
fig = plt.figure(figsize=(20,8))
n = 121
for score in SCORES:
    mp.plot_learning_curve(
        fig.add_subplot(n), clf, title, X, Y, scoring=score[0],
        ylim=(score[1], score[2]), cv=cv, n_jobs=-1,
        train_sizes=np.linspace(.1, 1.0, 10))
    n += 1


# ##  LinearSVC - Curva de validación

# In[15]:

PARAM_NAMES = ['estimator__C']
PARAM_RANGES = [[1, 10, 100, 1000]]
cv = ShuffleSplit(n_splits=10, test_size=0.5)
clf = MultiOutputClassifier(LinearSVC())
fig = plt.figure(figsize=(20, 8))
n = 121
for score in SCORES:
    ax = mp.plot_validation_curve(
        fig.add_subplot(n), clf, X, Y, param_name=PARAM_NAMES[0],
        param_range=PARAM_RANGES[0], cv=cv, scoring=score[0],
        ylim=(score[1], score[2]))
    ax.semilogx()
    n += 1


# ## LinearSVC - Resultados

# In[16]:

print('Normal')
clf = MultiOutputClassifier(LinearSVC())
display(mp.cross_val_report(clf, cv, X, Y, True))
print('Chained')
clf = ChainedMultiOutputClassifier(LinearSVC())
display(mp.cross_val_report(clf, cv, X, Y, True))


# In[17]:

CUSTOM2_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'Sd', 'BLZ_CSpeed',
              'Mean', 'Kurt', 'Skew', 'Max', 'Min']
X = scale(df[CUSTOM2_IN].values)
print('Normal')
clf = MultiOutputClassifier(LinearSVC())
display(mp.cross_val_report(clf, cv, X, Y, True))
print('Chained')
clf = ChainedMultiOutputClassifier(LinearSVC())
display(mp.cross_val_report(clf, cv, X, Y, True))


# In[18]:

CUSTOM_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'BLZ_CSpeed']
X = scale(df[CUSTOM_IN].values)
print('Normal')
clf = MultiOutputClassifier(LinearSVC())
display(mp.cross_val_report(clf, cv, X, Y, True))
print('Chained')
clf = ChainedMultiOutputClassifier(LinearSVC())
display(mp.cross_val_report(clf, cv, X, Y, True))


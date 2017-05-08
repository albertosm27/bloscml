
# coding: utf-8

# # Algoritmos Random Forest

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from multioutput_chained import ChainedMultiOutputClassifier
import ml_plots as mp
import scoring_functions as sf
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# In[3]:

SCORES = [(sf.balanced, 0.4, 1.01), (sf.brier, -2, 0)]
IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 
              'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1',
              'Q3', 'N_Streaks', 'BLZ_CRate', 'BLZ_CSpeed', 
              'BLZ_DSpeed', 'LZ4_CRate', 'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Zstd']
OUT_FILTER = ['Noshuffle', 'Shuffle', 'Bitshuffle'] 
OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_BLOCKS = ['Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128',
              'Block_256', 'Block_512', 'Block_1024', 'Block_2048']
OUT_OPTIONS = OUT_CODEC + OUT_FILTER + OUT_LEVELS + OUT_BLOCKS


# In[4]:

df = pd.read_csv('../data/training_data.csv', sep='\t')
X, Y = df[IN_OPTIONS].values, df[OUT_OPTIONS].values


# In[5]:

rfc = RandomForestClassifier(n_estimators=50, n_jobs=-1)


# ## RFC - Curvas de aprendizaje

# In[6]:

title = "Learning Curves (Random Forest)"
cv = ShuffleSplit(n_splits=10, test_size=0.1)
fig = plt.figure(figsize=(20,8))
n = 121
for score in SCORES:
    mp.plot_learning_curve(
        fig.add_subplot(n), rfc, title, X, Y, scoring=score[0],
        ylim=(score[1], score[2]), cv=cv, n_jobs=-1,
        train_sizes=np.linspace(.1, 1.0, 10))
    n += 1


# ## RFC - Curvas de validación

# In[7]:

PARAM_NAMES = ['n_estimators', 'max_features', 'max_depth']
PARAM_RANGES = [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
                [6, 8, 10, 12, 14, 16, 18, 20]]
cv = ShuffleSplit(n_splits=10, test_size=0.25)
for i in range(3):
    fig = plt.figure(figsize=(20, 8))
    n = 121
    for score in SCORES:
        mp.plot_validation_curve(
            fig.add_subplot(n), rfc, X, Y, param_name=PARAM_NAMES[i],
            param_range=PARAM_RANGES[i], cv=cv, scoring=score[0],
            ylim=(score[1], score[2]))
        n += 1


# ## RFC - Validación cruzada de hiperparámetros

# In[8]:

nested_clf = joblib.load(
    '../src/RFCnested_estimators_my_accuracy_scorer.pkl')
non_nested_clf = joblib.load(
    '../src/RFCnon_nested_estimators_my_accuracy_scorer.pkl')


# In[9]:

mp.print_nested_winners(nested_clf, non_nested_clf)


# In[10]:

nested_clf = joblib.load(
    '../src/RFCnested_estimators_my_brier_scorer.pkl')
non_nested_clf = joblib.load(
    '../src/RFCnon_nested_estimators_my_brier_scorer.pkl')
mp.print_nested_winners(nested_clf, non_nested_clf)


# In[11]:

del nested_clf
del non_nested_clf


# ## RFC - Resultados

# In[12]:

clf1 = RandomForestClassifier(
    n_estimators=20, max_features=10, max_depth=12, bootstrap=False,
    criterion='entropy')
clf2 = RandomForestClassifier(
    n_estimators=20, max_features=10, max_depth=12, bootstrap=True,
    criterion='gini')
clf3 = RandomForestClassifier(
    n_estimators=20, max_features=10, max_depth=12, bootstrap=True,
    criterion='entropy')
cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=1)


# In[13]:

print('Entropy')
mp.cross_val_report(clf1, cv, X, Y)


# In[14]:

print('Entropy bootstrap')
mp.cross_val_report(clf2, cv, X, Y)


# In[15]:

print('Gini bootstrap')
mp.cross_val_report(clf3, cv, X, Y)


# In[16]:

clf1.fit(X,Y)


# In[17]:

my_dict = dict(zip(clf1.feature_importances_, IN_OPTIONS))
NEW_IN = []
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))
    if elem > 0.01:
        NEW_IN.append(my_dict.get(elem))


# In[18]:

X = df[NEW_IN].values


# In[19]:

clf1 = RandomForestClassifier(
    n_estimators=20, max_features=10, max_depth=12, bootstrap=False,
    criterion='entropy')
mp.cross_val_report(clf1, cv, X, Y)


# In[20]:

clf1.fit(X, Y)


# In[21]:

my_dict = dict(zip(clf1.feature_importances_, NEW_IN))
NEW_IN2 = []
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))
    if elem > 0.025:
        NEW_IN2.append(my_dict.get(elem))


# In[22]:

X = df[NEW_IN2].values
clf1 = RandomForestClassifier(
    n_estimators=20, max_features=10, max_depth=12, bootstrap=False,
    criterion='entropy')
mp.cross_val_report(clf1, cv, X, Y)


# In[23]:

clf1.fit(X, Y)
my_dict = dict(zip(clf1.feature_importances_, NEW_IN2))
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))


# In[24]:

CUSTOM3_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'Sd', 'BLZ_CSpeed',
              'Mean', 'Kurt', 'Skew', 'N_Streaks', 'Max', 'Min']
X = df[CUSTOM3_IN].values
clf1 = RandomForestClassifier(
    n_estimators=20, max_features=8, max_depth=12, bootstrap=False,
    criterion='entropy')
display(mp.cross_val_report(clf1, cv, X, Y))
clf1.fit(X, Y)
my_dict = dict(zip(clf1.feature_importances_, CUSTOM3_IN))
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))


# In[25]:

CUSTOM2_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'Sd', 'BLZ_CSpeed',
              'Mean', 'Kurt', 'Skew', 'Max', 'Min']
X = df[CUSTOM2_IN].values
clf1 = RandomForestClassifier(
    n_estimators=20, max_features=8, max_depth=12, bootstrap=False,
    criterion='entropy')
display(mp.cross_val_report(clf1, cv, X, Y))
clf1.fit(X, Y)
my_dict = dict(zip(clf1.feature_importances_, CUSTOM2_IN))
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))


# In[26]:

CUSTOM_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'BLZ_CSpeed']
X = df[CUSTOM_IN].values
clf1 = RandomForestClassifier(
    n_estimators=20, max_features=4, max_depth=12, bootstrap=False,
    criterion='entropy')
display(mp.cross_val_report(clf1, cv, X, Y))
clf1.fit(X, Y)
my_dict = dict(zip(clf1.feature_importances_, CUSTOM_IN))
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))


# In[27]:

CUSTOM_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'LZ4_CRate', 'LZ4_CSpeed']
X = df[CUSTOM_IN].values
clf1 = RandomForestClassifier(
    n_estimators=20, max_features=4, max_depth=12, bootstrap=False,
    criterion='entropy')
display(mp.cross_val_report(clf1, cv, X, Y))
clf1.fit(X, Y)
my_dict = dict(zip(clf1.feature_importances_, CUSTOM_IN))
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))


# In[28]:

CUSTOM_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'Mean', 'Max', 'Min']
X = df[CUSTOM_IN].values
clf1 = RandomForestClassifier(
    n_estimators=20, max_features=4, max_depth=12, bootstrap=False,
    criterion='entropy')
display(mp.cross_val_report(clf1, cv, X, Y))
clf1.fit(X, Y)
my_dict = dict(zip(clf1.feature_importances_, CUSTOM_IN))
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))


# In[29]:

CUSTOM_IN = ['IN_CS', 'IN_DS', 'IN_CR']
X = df[CUSTOM_IN].values
clf1 = RandomForestClassifier(
    n_estimators=20, max_features=3, max_depth=12, bootstrap=False,
    criterion='entropy')
display(mp.cross_val_report(clf1, cv, X, Y))
clf1.fit(X, Y)
my_dict = dict(zip(clf1.feature_importances_, CUSTOM_IN))
for elem in np.sort(clf1.feature_importances_)[::-1]:
    print('%-12s --> %s' % (my_dict.get(elem), elem))


# ## MultiOutputs

# In[30]:

CUSTOM3_IN = ['IN_CS', 'IN_DS', 'IN_CR', 'BLZ_CRate', 'Sd', 'BLZ_CSpeed',
              'Mean', 'Kurt', 'Skew', 'N_Streaks', 'Max', 'Min']
X = df[CUSTOM3_IN].values
clf1 = ChainedMultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=20, max_features=8, max_depth=12, bootstrap=False,
        criterion='entropy'))
display(mp.cross_val_report(clf1, cv, X, Y, True))


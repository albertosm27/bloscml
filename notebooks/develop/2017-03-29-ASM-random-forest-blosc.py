
# coding: utf-8

# # Multioutput-Multiclass Random Forest Blosc

# ## Objetivos
# * Crear un algoritmo de arboles de decisión basado en bosques aleatorios utilizando scikit-learn.

# In[1]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas')


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
from sklearn.preprocessing import binarize 
from sklearn.preprocessing import OneHotEncoder

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# ## Importando los datos de entramiento
# Para ver como se crearon los datos de entrenamiento ir a [Training data generator](../deliver/training_data_generator.ipynb)

# In[3]:

df = pd.read_csv('../data/training_data.csv', sep='\t')


# ## Preprocesamiento entradas (extraer en training data generator o antes)

# In[4]:

df = df.assign(is_Table=binarize(df['Table'].values.reshape(-1,1), 0), 
               is_Columnar=binarize(df['Table'].values.reshape(-1,1), 1),
               is_Int=df['DType'].str.contains('int').astype(int),
               is_Float=df['DType'].str.contains('float').astype(int),
               is_String=(df['DType'].str.contains('S') | df['DType'].str.contains('U')).astype(int))
def aux_func(n):
    if n == 32 or n == 64:
        return n // 8
    else:
        return n
df['Type_Size'] = [aux_func(int(s)) for s in df['DType'].str[-2:]]


# ## Preprocesamiento salidas

# In[5]:

df = df.assign(Blosclz=(df['Codec'] == 'blosclz').astype(int),
               Lz4=(df['Codec'] == 'lz4').astype(int),
               Lz4hc=(df['Codec'] == 'lz4hc').astype(int),
               Snappy=(df['Codec'] == 'snappy').astype(int),
               Zstd=(df['Codec'] == 'zstd').astype(int),
               Shuffle=(df['Filter'] == 'shuffle').astype(int),
               Bitshuffle=(df['Filter'] == 'bitshuffle').astype(int))
enc_cl = OneHotEncoder()
enc_cl.fit(df['CL'].values.reshape(-1, 1))
new_cls = enc_cl.transform(df['CL'].values.reshape(-1, 1)).toarray()
enc_block = OneHotEncoder()
enc_block.fit(df['Block_Size'].values.reshape(-1, 1))
new_blocks = enc_block.transform(df['Block_Size'].values.reshape(-1, 1)).toarray()
block_sizes = [0, 8, 16, 32, 64, 128, 256, 512, 1024]
for i in range(9):
    cl_label = 'CL' + str(i+1)
    block_label = 'Block_' + str(block_sizes[i])
    df[cl_label] = new_cls[:, i]
    df[block_label] = new_blocks[:, i]
df['Block_2048'] = new_blocks[:, 9]


# In[6]:

IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_OPTIONS = ['Block_0', 'Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128', 'Block_256', 'Block_512', 'Block_1024', 'Block_2048',
               'Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd', 'Shuffle', 'Bitshuffle',
               'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']


# In[7]:

X, Y = df[IN_OPTIONS].values, df[OUT_OPTIONS].values


# In[8]:

from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# [DIFFER] thresholds randomness instead of most discriminative
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest)


# In[9]:

from sklearn import metrics
print(metrics.classification_report(Ytest, Ypred, digits=3, target_names=OUT_OPTIONS))


# In[10]:

clf.score(Xtest, Ytest)


# In[11]:

count = 0
for i in range(Ytest.shape[0]):
    if (Ytest[i,:] == Ypred[i,:]).all():
        count += 1
print(count/Ytest.shape[0])


# In[12]:

from sklearn.metrics import f1_score
f1_score(Ytest, Ypred, average='weighted')


# In[13]:

OUT_OPTIONS


# In[14]:

def my_score(Yreal, Ypred):
    score = 0
    for i in range(Yreal.shape[0]):
        if (Ytest[i,0:10] == Ypred[i,0:10]).all():
            score += 0.2
        if (Ytest[i,10:15] == Ypred[i,10:15]).all():
            score += 0.4
        if (Ytest[i,15:17] == Ypred[i,15:17]).all():
            score += 0.2
        if (Ytest[i,17:26] == Ypred[i,17:26]).all():
            score += 0.2
    return score/Yreal.shape[0]
my_score(Ytest, Ypred)


# Demasiada buena puntuación, busquemos algo más exigente.

# In[15]:

def my_score2(Yreal, Ypred):
    score = 0
    for i in range(Yreal.shape[0]):
        if (Ytest[i,0:10] == Ypred[i,0:10]).all() and (Ytest[i,17:26] == Ypred[i,17:26]).all():
            score += 0.5
        if (Ytest[i,10:15] == Ypred[i,10:15]).all() and (Ytest[i,15:17] == Ypred[i,15:17]).all():
            score += 0.5
    return score/Yreal.shape[0]
my_score2(Ytest, Ypred)


# In[17]:

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
X, Y = df[IN_OPTIONS].values, df[OUT_OPTIONS].values

param_grid = {'n_estimators': [100, 500],
              'criterion': ['gini', 'entropy'],
              'bootstrap': [True, False],
              'max_features': [1, 5, 10],
              'min_samples_leaf': [1, 5],
              'class_weight': [None, 'balanced']}

param_dist = {'n_estimators': [100, 200, 500],
              'criterion': ['gini', 'entropy'],
              'bootstrap': [True, False],
              'max_features': sp_randint(1, 10),
              'min_samples_leaf': sp_randint(1, 5),
              'class_weight': [None, 'balanced']}
ss = ShuffleSplit(n_splits=1, test_size=0.5)
rfc = RandomForestClassifier(n_jobs=-1)
grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ss, verbose=10, n_jobs=-1)
rgrid_rfc = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist, cv=ss, verbose=10, n_jobs=-1)


# In[22]:

grid_rfc.fit(X, Y)


# In[69]:

from IPython.display import HTML, display
score_param = []
for i in range(len(grid_rfc.cv_results_['mean_test_score'])):
    if grid_rfc.cv_results_['mean_test_score'][i] > 0.19:
        tup = (grid_rfc.cv_results_['mean_test_score'][i], grid_rfc.cv_results_['params'][i].items())
        score_param += [tup]
display(HTML(
    '<table><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in score_param)
        )
 ))


# In[102]:

dtc = grid_rfc.best_estimator_.estimators_[0]


# In[ ]:

import pydotplus
from IPython.display import Image
from sklearn import tree
dot_data = tree.export_graphviz(dtc, out_file=None, 
                         feature_names=IN_OPTIONS,  
                         class_names=OUT_OPTIONS,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())


# In[71]:

rgrid_rfc.fit(X, Y)


# ## Final del notebook
# Pasamos a desarrollar en la máquina remota de lineo, que es más potente, dejaremos un pequeño script lanzado y probaremos la persistencia

# In[20]:

from sklearn.externals import joblib
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              'criterion': ['gini', 'entropy'],
              'bootstrap': [True, False],
              'max_features': [10],
              'class_weight': [None, 'balanced']}
ss = ShuffleSplit(n_splits=5, test_size=0.5)
rfc = RandomForestClassifier(n_jobs=-1)
grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ss, verbose=1, n_jobs=-1)
grid_rfc.fit(X, Y)
joblib.dump(grid_rfc, 'grid_rfc_estimators.pkl')
# To load
# clf = joblib.load('filename.pkl') 


# In[55]:

data = []
indices = []
for i in range(len(grid_rfc.cv_results_['mean_test_score'])):
    if grid_rfc.cv_results_['mean_test_score'][i] > 0.20:
        row = np.append(grid_rfc.cv_results_['mean_test_score'][i], list(grid_rfc.cv_results_['params'][i].values()))
        data.append(row)
        indices += [i]
grid_best_df = pd.DataFrame(data=data,columns=['Score'] + list(grid_rfc.param_grid.keys()), index=indices)
grid_best_df['Score'] = grid_best_df['Score'].astype(float)
grid_best_df.sort(columns=['Score'], ascending=False)


# ## LEtZZZZ ROCKKKK!

# First we define three scoring functions properly

# In[58]:

def my_ponderated_scorer(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i,0:10] == ypred[i,0:10]).all():
            score += 0.2
        if (y[i,10:15] == ypred[i,10:15]).all():
            score += 0.4
        if (y[i,15:17] == ypred[i,15:17]).all():
            score += 0.2
        if (y[i,17:26] == ypred[i,17:26]).all():
            score += 0.2
    return score/y.shape[0]

def my_balanced_scorer(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i,0:10] == ypred[i,0:10]).all():
            score += 0.25
        if (y[i,10:15] == ypred[i,10:15]).all():
            score += 0.25
        if (y[i,15:17] == ypred[i,15:17]).all():
            score += 0.25
        if (y[i,17:26] == ypred[i,17:26]).all():
            score += 0.25
    return score/y.shape[0]

def my_2paired_scorer(predictor, X, y):
    ypred = predictor.predict(X)
    score = 0
    for i in range(y.shape[0]):
        if (y[i,0:10] == ypred[i,0:10]).all() and (y[i,17:26] == ypred[i,17:26]).all():
            score += 0.5
        if (y[i,10:15] == ypred[i,10:15]).all() and (y[i,15:17] == ypred[i,15:17]).all():
            score += 0.5
    return score/y.shape[0]


# And now we ROCK HARD!!!

# In[ ]:

param_grid = {'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90, 100],
              'max_depth': [None, 5, 10, 15, 20, 25, 30],
              'criterion': ['gini', 'entropy'],
              'bootstrap': [True, False],
              'max_features': [5, 7, 10, 15, 20],
              'class_weight': [None, 'balanced']}
ss = ShuffleSplit(n_splits=10, test_size=0.5)
rfc = RandomForestClassifier(n_jobs=-1)
scores = [None, my_balanced_scorer, my_ponderated_scorer, my_2paired_scorer]
for score in scores:
    grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ss, verbose=1, n_jobs=-1, scoring=score)
    grid_rfc.fit(X, Y)
    if score == None:
        joblib.dump(grid_rfc, 'grid_rfc_depth_default_scorer.pkl')
    else:
        joblib.dump(grid_rfc, 'grid_rfc_depth_default_' + score.__name__ + '.pkl')
        


# In[ ]:




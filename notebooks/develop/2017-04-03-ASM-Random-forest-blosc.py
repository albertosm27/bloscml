
# coding: utf-8

# # Primeros pickle

# In[2]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas, scikit-learn')


# In[4]:

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

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# In[14]:

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


# In[40]:

grid = joblib.load('grid_rfc_depth_default_scorer.pkl.gz')


# In[11]:

grid.param_grid


# In[41]:

grid.best_score_


# In[10]:

data = []
indices = []
for i in range(len(grid.cv_results_['mean_test_score'])):
    if grid.cv_results_['mean_test_score'][i] > 0.23:
        row = np.append(grid.cv_results_['mean_test_score'][i], list(grid.cv_results_['params'][i].values()))
        data.append(row)
        indices += [i]
grid_best_df = pd.DataFrame(data=data,columns=['Score'] + list(grid.param_grid.keys()), index=indices)
grid_best_df['Score'] = grid_best_df['Score'].astype(float)
grid_best_df.sort(columns=['Score'], ascending=False)


# In[15]:

grid = joblib.load('grid_rfc_depth_default_my_balanced_scorer.pkl.gz')


# In[16]:

grid.param_grid


# In[17]:

grid.best_score_


# In[19]:

data = []
indices = []
for i in range(len(grid.cv_results_['mean_test_score'])):
    if grid.cv_results_['mean_test_score'][i] > 0.67:
        row = np.append(grid.cv_results_['mean_test_score'][i], list(grid.cv_results_['params'][i].values()))
        data.append(row)
        indices += [i]
grid_best_df = pd.DataFrame(data=data,columns=['Score'] + list(grid.param_grid.keys()), index=indices)
grid_best_df['Score'] = grid_best_df['Score'].astype(float)
grid_best_df.sort(columns=['Score'], ascending=False)


# In[44]:

grid = joblib.load('grid_rfc_depth_default_my_ponderated_scorer.pkl.gz')


# In[21]:

grid.param_grid


# In[45]:

grid.best_score_


# In[24]:

data = []
indices = []
for i in range(len(grid.cv_results_['mean_test_score'])):
    if grid.cv_results_['mean_test_score'][i] > 0.72:
        row = np.append(grid.cv_results_['mean_test_score'][i], list(grid.cv_results_['params'][i].values()))
        data.append(row)
        indices += [i]
grid_best_df = pd.DataFrame(data=data,columns=['Score'] + list(grid.param_grid.keys()), index=indices)
grid_best_df['Score'] = grid_best_df['Score'].astype(float)
grid_best_df.sort(columns=['Score'], ascending=False)


# In[25]:

grid = joblib.load('grid_rfc_depth_default_my_2paired_scorer.pkl.gz')


# In[26]:

grid.param_grid


# In[27]:

grid.best_score_


# In[28]:

data = []
indices = []
for i in range(len(grid.cv_results_['mean_test_score'])):
    if grid.cv_results_['mean_test_score'][i] > 0.56:
        row = np.append(grid.cv_results_['mean_test_score'][i], list(grid.cv_results_['params'][i].values()))
        data.append(row)
        indices += [i]
grid_best_df = pd.DataFrame(data=data,columns=['Score'] + list(grid.param_grid.keys()), index=indices)
grid_best_df['Score'] = grid_best_df['Score'].astype(float)
grid_best_df.sort(columns=['Score'], ascending=False)


# In[46]:

my_ponderated_scorer(grid.best_estimator_, X, Y)


# In[29]:

df = pd.read_csv('../data/training_data.csv', sep='\t')


# In[31]:

from sklearn.preprocessing import binarize 
from sklearn.preprocessing import OneHotEncoder
df = df.assign(is_Table=binarize(df['Table'].values.reshape(-1,1), 0), 
               is_Columnar=binarize(df['Table'].values.reshape(-1,1), 1),
               is_Int=df['DType'].str.contains('int').astype(int),
               is_Float=df['DType'].str.contains('float').astype(int),
               is_String=(df['DType'].str.contains('S') | df['DType'].str.contains('U')).astype(int))
import re
def aux_func(s):
    n = int(re.findall('\d+', s)[0])
    isNum = re.findall('int|float', s)
    if len(isNum) > 0:
        return n // 8
    else:
        return n
df['Type_Size'] = [aux_func(s) for s in df['DType']]


# In[32]:

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


# In[33]:

IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_OPTIONS = ['Block_0', 'Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128', 'Block_256', 'Block_512', 'Block_1024', 'Block_2048',
               'Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd', 'Shuffle', 'Bitshuffle',
               'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd']


# In[36]:

X, Y = df[IN_OPTIONS].values, df[OUT_CODEC].values


# In[49]:

from sklearn.model_selection import ShuffleSplit
param_grid = {'n_estimators': [60, 70, 80, 90, 100],
              'max_depth': [None, 15, 20, 25, 30],
              'criterion': ['entropy'],
              'bootstrap': [False],
              'max_features': [15, 20, 25],
              'class_weight': [None]}
ss = ShuffleSplit(n_splits=10, test_size=0.25)
rfc = RandomForestClassifier(n_jobs=-1)
grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ss, verbose=1, n_jobs=-1)


# In[51]:

grid_rfc.fit(X, Y)


# In[ ]:




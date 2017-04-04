
# coding: utf-8

# # Primeros pickle

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

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# In[3]:

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


# In[4]:

grid = joblib.load('grid_rfc_depth_default_scorer.pkl.gz')


# In[5]:

grid.param_grid


# In[6]:

grid.best_score_


# In[7]:

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


# In[8]:

grid = joblib.load('grid_rfc_depth_default_my_balanced_scorer.pkl.gz')


# In[9]:

grid.param_grid


# In[10]:

grid.best_score_


# In[11]:

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


# In[12]:

grid = joblib.load('grid_rfc_depth_default_my_ponderated_scorer.pkl.gz')


# In[13]:

grid.param_grid


# In[14]:

grid.best_score_


# In[15]:

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


# In[16]:

grid = joblib.load('grid_rfc_depth_default_my_2paired_scorer.pkl.gz')


# In[17]:

grid.param_grid


# In[18]:

grid.best_score_


# In[19]:

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


# In[21]:

df = pd.read_csv('../data/training_data.csv', sep='\t')


# In[22]:

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


# In[23]:

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


# In[24]:

IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_OPTIONS = ['Block_0', 'Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128', 'Block_256', 'Block_512', 'Block_1024', 'Block_2048',,
               'Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd', 'Shuffle', 'Bitshuffle',
               'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd']


# In[142]:

X, Y = df[IN_OPTIONS].values, df[OUT_CODEC].values


# In[26]:

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


# In[27]:

grid_rfc.fit(X, Y)


# In[28]:

grid_rfc.best_score_


# In[29]:

grid_rfc.best_params_


# In[31]:

grid = grid_rfc
data = []
indices = []
for i in range(len(grid.cv_results_['mean_test_score'])):
    if grid.cv_results_['mean_test_score'][i] > 0.96:
        row = np.append(grid.cv_results_['mean_test_score'][i], list(grid.cv_results_['params'][i].values()))
        data.append(row)
        indices += [i]
grid_best_df = pd.DataFrame(data=data,columns=['Score'] + list(grid.param_grid.keys()), index=indices)
grid_best_df['Score'] = grid_best_df['Score'].astype(float)
grid_best_df.sort(columns=['Score'], ascending=False)


# In[140]:

rfc = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)


# In[146]:

from sklearn.model_selection import cross_val_score
ss = ShuffleSplit(n_splits=10, test_size=0.5)
scores = cross_val_score(rfc, X, Y, cv=ss)


# In[147]:

scores


# In[148]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[93]:

from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
clf = make_pipeline(preprocessing.StandardScaler(), rfc)
scores = cross_val_score(clf, X, Y, cv=ss)


# In[94]:

scores


# In[95]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[132]:

param_grid = {'n_estimators': [80],
              'max_depth': [20],
              'criterion': ['entropy'],
              'bootstrap': [False],
              'max_features': [15],
              'class_weight': [None]}
grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ss, verbose=1, n_jobs=-1)
grid_rfc.fit(X, Y)


# In[133]:

grid_rfc.score(X, Y)


# In[108]:

OUT_FILTER = ['Shuffle', 'Bitshuffle']
aux = np.empty((X.shape[0], X.shape[1]+Y.shape[1]))
aux[:,:X.shape[1]] = X
aux[:,X.shape[1]:X.shape[1]+Y.shape[1]] = grid_rfc.predict(X)


# In[109]:

X = aux
Y = df[OUT_FILTER].values


# In[110]:

ss = ShuffleSplit(n_splits=10, test_size=0.25)
scores = cross_val_score(rfc, X, Y, cv=ss)


# In[111]:

scores


# In[112]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[113]:

param_grid = {'n_estimators': [80],
              'max_depth': [20],
              'criterion': ['entropy'],
              'bootstrap': [False],
              'max_features': [15],
              'class_weight': [None]}
grid_rfc2 = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ss, verbose=1, n_jobs=-1)
grid_rfc2.fit(X, Y)


# In[114]:

grid_rfc2.best_score_


# In[115]:

grid_rfc2.score(X, Y)


# In[116]:

OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
aux = np.empty((X.shape[0], X.shape[1]+Y.shape[1]))
aux[:,:X.shape[1]] = X
aux[:,X.shape[1]:X.shape[1]+Y.shape[1]] = grid_rfc2.predict(X)


# In[117]:

X = aux
Y = df[OUT_LEVELS].values


# In[118]:

scores = cross_val_score(rfc, X, Y, cv=ss)


# In[119]:

scores


# In[120]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[121]:

param_grid = {'n_estimators': [80],
              'max_depth': [20],
              'criterion': ['entropy'],
              'bootstrap': [False],
              'max_features': [15],
              'class_weight': [None]}
grid_rfc3 = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ss, verbose=1, n_jobs=-1)
grid_rfc3.fit(X, Y)


# In[122]:

grid_rfc3.best_score_


# In[123]:

grid_rfc3.score(X, Y)


# In[124]:

OUT_BLOCKS = ['Block_0', 'Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128', 'Block_256', 'Block_512', 'Block_1024', 'Block_2048']
aux = np.empty((X.shape[0], X.shape[1] + Y.shape[1]))
aux[:,:X.shape[1]] = X
aux[:,X.shape[1]:X.shape[1] + Y.shape[1]] = grid_rfc3.predict(X)


# In[126]:

X = aux
Y = df[OUT_BLOCKS].values


# In[128]:

param_grid = {'n_estimators': [80],
              'max_depth': [20],
              'criterion': ['entropy'],
              'bootstrap': [False],
              'max_features': [15],
              'class_weight': [None]}
grid_rfc3 = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ss, verbose=1, n_jobs=-1)
grid_rfc3.fit(X, Y)


# In[129]:

grid_rfc3.best_score_


# In[130]:

grid_rfc3.score(X, Y)


# Así pues surge un claro problema, vamos a dividir nuestros datos de entrenamiento de forma equitativa.

# In[227]:

chunks_df = df.drop_duplicates(subset=['DataSet', 'Table', 'Chunk_Number'])
train_data = pd.DataFrame(columns=df.columns.values)
test_data = pd.DataFrame(columns=df.columns.values)
spliter = ShuffleSplit(n_splits=1, test_size=0.4)
for i_train, i_test in spliter.split(chunks_df.values):
    aux_train = chunks_df.iloc[i_train][['DataSet', 'Table', 'Chunk_Number']]
    aux_test = chunks_df.iloc[i_test][['DataSet', 'Table', 'Chunk_Number']]
    for index, row in aux_train.iterrows():
        train_data = train_data.append(df[(df.DataSet == row['DataSet']) & (df.Table == row['Table']) &
                             (df.Chunk_Number == row['Chunk_Number'])])
    for index, row in aux_test.iterrows():
        test_data = test_data.append(df[(df.DataSet == row['DataSet']) & (df.Table == row['Table']) &
                             (df.Chunk_Number == row['Chunk_Number'])])


# In[246]:

X, Y = train_data[IN_OPTIONS].values, train_data[OUT_FILTER].values


# In[247]:

Xtest, Ytest = test_data[IN_OPTIONS].values, test_data[OUT_FILTER].values


# In[230]:

rfc = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)


# In[231]:

rfc.fit(X, Y)


# In[232]:

rfc.score(Xtest, Ytest)


# In[233]:

OUT_FILTER


# In[234]:

aux = np.empty((X.shape[0], X.shape[1] + Y.shape[1]))
aux[:,:X.shape[1]] = X
aux[:,X.shape[1]:X.shape[1] + Y.shape[1]] = rfc.predict(X)

aux2 = np.empty((Xtest.shape[0], Xtest.shape[1] + Ytest.shape[1]))
aux2[:,:Xtest.shape[1]] = Xtest
aux2[:,Xtest.shape[1]:Xtest.shape[1] + Ytest.shape[1]] = rfc.predict(Xtest)


# In[235]:

X, Xtest = aux, aux2
Y, Ytest = train_data[OUT_FILTER].values, test_data[OUT_FILTER].values


# In[236]:

rfc2 = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)
rfc2.fit(X,Y)


# In[237]:

rfc2.score(Xtest, Ytest)


# In[238]:

OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
aux = np.empty((X.shape[0], X.shape[1] + Y.shape[1]))
aux[:,:X.shape[1]] = X
aux[:,X.shape[1]:X.shape[1] + Y.shape[1]] = rfc2.predict(X)

aux2 = np.empty((Xtest.shape[0], Xtest.shape[1] + Ytest.shape[1]))
aux2[:,:Xtest.shape[1]] = Xtest
aux2[:,Xtest.shape[1]:Xtest.shape[1] + Ytest.shape[1]] = rfc2.predict(Xtest)


# In[239]:

X, Xtest = aux, aux2
Y, Ytest = train_data[OUT_LEVELS].values, test_data[OUT_LEVELS].values


# In[240]:

rfc3 = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)
rfc3.fit(X,Y)


# In[241]:

rfc3.score(Xtest, Ytest)


# In[242]:

OUT_BLOCKS = ['Block_0', 'Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128', 'Block_256', 'Block_512', 'Block_1024', 'Block_2048']
aux = np.empty((X.shape[0], X.shape[1] + Y.shape[1]))
aux[:,:X.shape[1]] = X
aux[:,X.shape[1]:X.shape[1] + Y.shape[1]] = rfc3.predict(X)

aux2 = np.empty((Xtest.shape[0], Xtest.shape[1] + Ytest.shape[1]))
aux2[:,:Xtest.shape[1]] = Xtest
aux2[:,Xtest.shape[1]:Xtest.shape[1] + Ytest.shape[1]] = rfc3.predict(Xtest)


# In[243]:

X, Xtest = aux, aux2
Y, Ytest = train_data[OUT_BLOCKS].values, test_data[OUT_BLOCKS].values


# In[244]:

rfc4 = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)
rfc4.fit(X,Y)


# In[245]:

rfc4.score(Xtest, Ytest)


# In[ ]:




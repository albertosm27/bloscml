
# coding: utf-8

# # Reordenando ideas

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

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# In[3]:

df = pd.read_csv('../data/training_data.csv', sep='\t')


# In[7]:

IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
IN2_OPTIONS = ['IN_1', 'IN_2', 'IN_3', 'IN_4', 'IN_5', 'IN_6', 'IN_7', 'is_Array', 'is_Table', 'is_Columnar', 'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_CODEC = ['Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd']
OUT_FILTER = ['Noshuffle', 'Shuffle', 'Bitshuffle'] 
OUT_LEVELS = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8', 'CL9']
OUT_BLOCKS = ['Block_8', 'Block_16', 'Block_32', 'Block_64', 'Block_128', 'Block_256', 'Block_512', 'Block_1024', 'Block_2048']
OUT_OPTIONS = OUT_CODEC + OUT_FILTER + OUT_LEVELS + OUT_BLOCKS


# In[10]:

df = df.assign(IN_1=((df.IN_CR == 1) & (df.IN_CS == 0) & (df.IN_DS == 0)).astype(int),
          IN_2=((df.IN_CR == 0) & (df.IN_CS == 1) & (df.IN_DS == 0)).astype(int),
          IN_3=((df.IN_CR == 0) & (df.IN_CS == 0) & (df.IN_DS == 1)).astype(int),
          IN_4=((df.IN_CR == 1) & (df.IN_CS == 1) & (df.IN_DS == 0)).astype(int),
          IN_5=((df.IN_CR == 1) & (df.IN_CS == 0) & (df.IN_DS == 1)).astype(int),
          IN_6=((df.IN_CR == 0) & (df.IN_CS == 1) & (df.IN_DS == 1)).astype(int),
          IN_7=((df.IN_CR == 1) & (df.IN_CS == 1) & (df.IN_DS == 1)).astype(int),
          is_Array=(df.Table == 0).astype(int), is_Table=(df.Table == 1).astype(int))
df['Noshuffle'] = (df['Filter'] == 'noshuffle').astype(int)


# In[11]:

X, Y = df[IN2_OPTIONS].values, df[OUT_OPTIONS].values


# In[14]:

ss = ShuffleSplit(n_splits=10, test_size=0.25)
rfc = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)
scores = cross_val_score(rfc, X, Y, cv=ss)


# In[15]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[18]:

X, Y = df[IN_OPTIONS].values, df[OUT_CODEC + OUT_FILTER + OUT_LEVELS].values
ss = ShuffleSplit(n_splits=10, test_size=0.25)
rfc = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)
scores = cross_val_score(rfc, X, Y, cv=ss)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[19]:

X, Y = df[IN_OPTIONS].values, df[OUT_BLOCKS].values
ss = ShuffleSplit(n_splits=10, test_size=0.25)
rfc = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)
scores = cross_val_score(rfc, X, Y, cv=ss)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[131]:

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


# In[132]:

X, Y = train_data[IN_OPTIONS].values, train_data[OUT_OPTIONS].values


# In[133]:

Xtest, Ytest = test_data[IN_OPTIONS].values, test_data[OUT_OPTIONS].values


# In[134]:

rfc.fit(X, Y)


# In[65]:

rfc.predict_proba(Xtest)[0][:,1]


# In[124]:

prob = rfc.predict_proba(Xtest)


# In[126]:

codec_probs = []
filter_probs = []
clevel_probs = []
bloc_probs = []
for i in range(len(prob[0])):
    codec_probs.append([prob[j][i][1] for j in range(5)])
    filter_probs.append([prob[j][i][1] for j in range(5, 8)])
    clevel_probs.append([prob[j][i][1] for j in range(8, 17)])
    bloc_probs.append([prob[j][i][1] for j in range(17, 26)])


# In[46]:

clf_score = brier_score_loss(Ytest[:,0], prob)
print("No calibration: %1.3f" % clf_score)


# In[128]:

def my_brier_scorer(predictor, X, y):
    probs = predictor.predict_proba(X)
    sorted_probs = []
    score = 0
    for i in range(len(prob[0])):
        sorted_probs.append([prob[j][i][1] for j in range(26)])
    for i in range(y.shape[0]):
        aux = np.square(sorted_probs[i] - y[i])
        score += np.mean(aux[0:5]) + np.mean(aux[5:8]) + np.mean(aux[8:17]) + np.mean(aux[17:26])
    return score/y.shape[0]


# In[136]:

ss = ShuffleSplit(n_splits=10, test_size=0.25)


# In[137]:

cross_val_score(rfc, X, Y, cv=ss, scoring=my_brier_scorer)


# In[8]:

df['Noshuffle'] = (df['Filter'] == 'noshuffle').astype(int)


# In[129]:

X, Y = df[IN_OPTIONS].values, df[OUT_FILTER2].values
ss = ShuffleSplit(n_splits=100, test_size=0.25)
rfc = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)
scores = cross_val_score(rfc, X, Y, cv=ss)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[59]:

X, Y = df[IN_OPTIONS].values, df[OUT_FILTER].values
ss = ShuffleSplit(n_splits=100, test_size=0.25)
rfc = RandomForestClassifier(n_estimators=70, bootstrap=False, class_weight=None, criterion='entropy',
                             max_features=15, max_depth=20, n_jobs=-1)
scores = cross_val_score(rfc, X, Y, cv=ss)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:




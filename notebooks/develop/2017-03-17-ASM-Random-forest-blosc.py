
# coding: utf-8

# # Multiclass Random Forest Blosc

# ## Objetivos
# * **[Dead End]** Crear un algoritmo de arboles de decisión basado en bosques aleatorios utilizando scikit-learn.

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

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# ## Importando los datos de entramiento
# Para ver como se crearon los datos de entrenamiento ir a [Training data generator](../deliver/training_data_generator.ipynb)

# In[3]:

df = pd.read_csv('../data/training_data.csv', sep='\t')


# In[4]:

df.head()


# In[5]:

# INPUT OPTIONS TUNING
from sklearn.preprocessing import binarize 
df = df.assign(is_Table=binarize(df['Table'].values.reshape(-1,1), 0), 
               is_Columnar=binarize(df['Table'].values.reshape(-1,1), 1),
               is_Int=df['DType'].str.contains('int').astype(int),
               is_Float=df['DType'].str.contains('float').astype(int),
               is_String=(df['DType'].str.contains('S') | df['DType'].str.contains('U')).astype(int))


# In[6]:

def aux_func(n):
    if n == 32 or n == 64:
        return n // 8
    else:
        return n
df['Type_Size'] = [aux_func(int(s)) for s in df['DType'].str[-2:]]


# In[7]:

# OUTPUT OPTIONS TUNING
df = df.assign(Blosclz=(df['Codec'] == 'blosclz').astype(int),
               Lz4=(df['Codec'] == 'lz4').astype(int),
               Lz4hc=(df['Codec'] == 'lz4hc').astype(int),
               Snappy=(df['Codec'] == 'snappy').astype(int),
               Zstd=(df['Codec'] == 'zstd').astype(int),
               Shuffle=(df['Filter'] == 'shuffle').astype(int),
               Bitshuffle=(df['Filter'] == 'bitshuffle').astype(int))
               #CL1=df['CL'] == 1,
               #CL2=df['CL'] == 2,
               #CL3=df['CL'] == 3,
               #CL4=df['CL'] == 4,
               #CL5=df['CL'] == 5,
               #CL6=df['CL'] == 6,
               #CL7=df['CL'] == 7,
               #CL8=df['CL'] == 8,
               #CL9=df['CL'] == 9,
               #BS0=df['Block_Size'] == 0,
               #BS16=df['Block_Size'] == 16,
               #BS32=df['Block_Size'] == 32,
               #BS64=df['Block_Size'] == 64,
               #BS128=df['Block_Size'] == 128,
               #BS256=df['Block_Size'] == 256,
               #BS512=df['Block_Size'] == 512,
               #BS1024=df['Block_Size'] == 1024,
               #BS2048=df['Block_Size'] == 2048
#INTS = ['IN_CR', 'IN_CS', 'IN_DS', 'CL', 'Block_Size']
#df[INTS] = df[INTS].apply(pd.to_numeric, args=('ignore', 'integer'))


# In[8]:

from sklearn.preprocessing import OneHotEncoder
# OUTPUT OPTIONS TUNING
enc_cl = OneHotEncoder()
enc_cl.fit(df['CL'].values.reshape(-1, 1))
new_cls = enc_cl.transform(df['CL'].values.reshape(-1, 1)).toarray()
enc_block = OneHotEncoder()
enc_block.fit(df['Block_Size'].values.reshape(-1, 1))
new_blocks = enc_block.transform(df['Block_Size'].values.reshape(-1, 1)).toarray()

for i in range(9):
    cl_label = 'CL' + str(i)
    block_label = 'Block' + str(i)
    df[cl_label] = new_cls[:, i]
    df[block_label] = new_blocks[:, i]
df['Block9'] = new_blocks[:, 9]


# In[9]:

# SINGLE OUTPUT OPTION TUNING
df['Classes'] = df['Codec'] + '-' + df['Filter'] + '-' + df['CL'].apply(str)+ '-' + df['Block_Size'].apply(str)
class_map = df.drop_duplicates(subset=['Classes'])['Classes'].to_dict()
class_map = {el:i for i, el in enumerate(class_map.values()) }
df['Classes_ID'] = df['Classes'].apply(class_map.get)


# In[10]:

#CHUNK_FEATURES = ["Table", "DType", "Chunk_Size", "Mean", "Median", "Sd", "Skew", "Kurt", "Min", "Max", "Q1", "Q3", "N_Streaks"]
IN_OPTIONS = ['IN_CR', 'IN_CS', 'IN_DS', 'is_Table', 'is_Columnar', 'is_Int', 'is_Float', 'is_String', 'Type_Size', 'Chunk_Size',
              'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3', 'BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
              'LZ4_CSpeed', 'LZ4_DSpeed']
OUT_OPTIONS = ['Block0', 'Block1', 'Block2', 'Block3', 'Block4', 'Block5', 'Block6', 'Block7', 'Block8', 'Block9',
               'Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd', 'Shuffle', 'Bitshuffle',
               'CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7', 'CL8']
#OUT_OPTIONS = ['Block_Size', 'Blosclz', 'Lz4', 'Lz4hc', 'Snappy', 'Zstd',
#               'Shuffle', 'Bitshuffle', 'CL']


# In[11]:

X, Y = df[IN_OPTIONS].values, df[['Classes_ID']].values.ravel()


# In[ ]:




# In[12]:

from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
# [DIFFER] thresholds randomness instead of most discriminative
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest)


# In[13]:

from sklearn import metrics
print(metrics.classification_report(Ytest, Ypred, digits=3))


# Insuficientes datos para plantear el problema como multiclass, además los datos de entrenamiento configurados de esta forma no están nada balanceados por lo que aunque tengamos suficientes datos seguiría siendo complicado obtener buenos resultados por este camino.

# In[ ]:




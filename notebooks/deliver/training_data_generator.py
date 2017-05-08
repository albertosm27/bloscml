
# coding: utf-8

# # Generador de los datos de entrenamiento

# ## Objetivos del análisis
# * Extraer el data frame final con los datos preparados para entrenar algoritmos machine learning.

# In[1]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, pandas')


# In[2]:

import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.3f}'.format


# In[3]:

CHUNK_ID = ["Filename", "DataSet", "Table", "Chunk_Number"]
CHUNK_FEATURES = ["Table", "DType", "Chunk_Size", "Mean", "Median", 
                  "Sd", "Skew", "Kurt", "Min", "Max", "Q1", "Q3",
                  "N_Streaks"]
OUT_OPTIONS = ["Block_Size", "Codec", "Filter", "CL"]
TEST_FEATURES = ["CRate", "CSpeed", "DSpeed"]
COLS = ["Filename" , "DataSet", "Chunk_Number"] + CHUNK_FEATURES 
       + OUT_OPTIONS + TEST_FEATURES
IN_TESTS = ['BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate',
            'LZ4_CSpeed', 'LZ4_DSpeed']
IN_USER = ['IN_CR', 'IN_CS', 'IN_DS']


# In[4]:

df = pd.read_csv('../data/blosc_test_data_final.csv.gz', sep='\t')
my_df = df[(df.CL != 0) & (df.CRate > 1.1)]


# ## Extracción de los datos por chunk

# In[5]:

get_ipython().run_cell_magic('time', '', '# DATAFRAME WITH DISTINCT CHUNKS\nchunks_df = my_df.drop_duplicates(subset=CHUNK_ID)\nprint("%d rows" % chunks_df.shape[0])\nchunk_tests_list = []\n# FOR EACH CHUNK\nfor index, row in chunks_df.iterrows():\n    # DATAFRAME WITH CHUNK TESTS\n    chunk_tests_list.append(my_df[(my_df.Filename == row["Filename"]) &\n                               (my_df.DataSet == row["DataSet"]) &\n                               (my_df.Table == row["Table"]) &\n                               (my_df.Chunk_Number == row["Chunk_Number"])])')


# ## Selección de opciones para cada chunk

# In[6]:

get_ipython().run_cell_magic('time', '', "training_df = pd.DataFrame()\nfor chunk_test in chunk_tests_list:\n    # EXTRACT MAX MIN AND SOME AUX MAX INDICES\n    i_max_crate, i_max_c_speed, i_max_d_speed = \\\n        chunk_test['CRate'].idxmax(), chunk_test['CSpeed'].idxmax(),\\\n        chunk_test['DSpeed'].idxmax()\n    max_crate, max_c_speed, max_d_speed = \\\n        (chunk_test.ix[i_max_crate]['CRate'], \n         chunk_test.ix[i_max_c_speed]['CSpeed'],\n         chunk_test.ix[i_max_d_speed]['DSpeed'])\n\n    min_crate, min_c_speed, min_d_speed = (chunk_test['CRate'].min(),\n                                           chunk_test['CSpeed'].min(),\n                                           chunk_test['DSpeed'].min())\n    # NORMALIZED COLUMNS\n    chunk_test = chunk_test.assign(N_CRate=(chunk_test['CRate'] \n                                            - min_crate) \n                                            / (max_crate - min_crate),\n                                   N_CSpeed=(chunk_test['CSpeed'] \n                                            - min_c_speed) \n                                            / (max_c_speed - min_c_speed),\n                                   N_DSpeed=(chunk_test['DSpeed'] \n                                            - min_d_speed) \n                                            / (max_d_speed - min_d_speed))\n    # DISTANCE FUNC COLUMNS\n    chunk_test = chunk_test.assign(Distance_1=(chunk_test['N_CRate'] - 1)**2\n                                   + (chunk_test['N_CSpeed'] - 1)**2,\n                                   Distance_2=(chunk_test['N_CRate'] - 1)**2\n                                   + (chunk_test['N_DSpeed'] - 1) ** 2,\n                                   Distance_3=(chunk_test['N_CRate'] - 1)**2\n                                   + (chunk_test['N_DSpeed'] - 1)**2\n                                   + (chunk_test['N_CSpeed'] - 1)**2,\n                                   Distance_4=(chunk_test['N_CSpeed'] - 1)**2\n                                   + (chunk_test['N_DSpeed'] - 1) ** 2)\n    # BALANCED INDICES\n    i_balanced_c_speed, i_balanced_d_speed,\n    i_balanced, i_balanced_speeds = (chunk_test['Distance_1'].idxmin(),\n                                     chunk_test['Distance_2'].idxmin(),\n                                     chunk_test['Distance_3'].idxmin(),\n                                     chunk_test['Distance_4'].idxmin())\n    indices = [i_max_d_speed, i_max_c_speed, i_balanced_speeds, \n               i_max_crate, i_balanced_d_speed, i_balanced_c_speed,\n               i_balanced]\n    # TYPE FILTER FOR LZ_DATA\n    d_type = chunk_test.iloc[0]['DType']\n    filter_name = 'noshuffle'\n    if 'float' in d_type or 'int' in d_type:\n        filter_name = 'shuffle'\n    aux = df[(df.CL == 1) & (df.Block_Size == 0) &\n             (df.Filter == filter_name) &\n             (df.Filename == chunk_test.iloc[0]['Filename']) &\n             (df.DataSet == chunk_test.iloc[0]['DataSet']) &\n             (df.Table == chunk_test.iloc[0]['Table']) & \n             (df.Chunk_Number == chunk_test.iloc[0]['Chunk_Number'])]\n    lz_data = np.append(aux[aux.Codec == 'blosclz'][TEST_FEATURES].values[0],\n                        aux[aux.Codec == 'lz4'][TEST_FEATURES].values[0])\n    # APPEND ROWS TO TRAINING DATA FRAME\n    for i in range(len(indices)):\n        in_1, r = divmod((i+1), 4)\n        in_2, in_3 = divmod(r, 2)\n        training_df = training_df.append(\n                        dict(zip(COLS + IN_TESTS + IN_USER,\n                        np.append(np.append(chunk_test.ix[indices[i]][COLS].values,\n                        lz_data),\n                        [in_1, in_2, in_3]))),\n                        ignore_index=True)")


# ## Algunas comprobaciones

# In[7]:

print('DISTINCT MAX RATE')
distinct_max_rate = training_df[(training_df.IN_CR == 1) &
                                (training_df.IN_CS == 0) & 
                                (training_df.IN_DS == 0)]\
                    .drop_duplicates(subset=OUT_OPTIONS)
                    [OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_max_rate.shape[0])
display(distinct_max_rate.head())
print('DISTINCT MAX C.SPEED')
distinct_max_c_speed = training_df[(training_df.IN_CR == 0) & 
                                   (training_df.IN_CS == 1) &
                                   (training_df.IN_DS == 0)]\
                       .drop_duplicates(subset=OUT_OPTIONS)
                       [OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_max_c_speed.shape[0])
display(distinct_max_c_speed.head())
print('DISTINCT MAX D.SPEED')
distinct_max_d_speed = training_df[(training_df.IN_CR == 0) &
                                   (training_df.IN_CS == 0) &
                                   (training_df.IN_DS == 1)]\
                      .drop_duplicates(subset=OUT_OPTIONS)
                      [OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_max_d_speed.shape[0])
display(distinct_max_d_speed.head())
print('DISTINCT BALANCED CSPEED')
distinct_balanced_c_speed = training_df[(training_df.IN_CR == 1) &
                                        (training_df.IN_CS == 1) &
                                        (training_df.IN_DS == 0)]\
                            .drop_duplicates(subset=OUT_OPTIONS)
                            [OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_balanced_c_speed.shape[0])
display(distinct_balanced_c_speed.head())
print('DISTINCT BALANCED DSPEED')
distinct_balanced_d_speed = training_df[(training_df.IN_CR == 1) &
                                        (training_df.IN_CS == 0) &
                                        (training_df.IN_DS == 1)]\
                            .drop_duplicates(subset=OUT_OPTIONS)
                            [OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_balanced_d_speed.shape[0])
display(distinct_balanced_d_speed.head())
print('DISTINCT BALANCED SPEED')
distinct_balanced_speed = training_df[(training_df.IN_CR == 0) &
                                      (training_df.IN_CS == 1) &
                                      (training_df.IN_DS == 1)]\
                          .drop_duplicates(subset=OUT_OPTIONS)
                          [OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_balanced_speed.shape[0])
display(distinct_balanced_speed.head())
print('DISTINCT BALANCED')
distinct_balanced = training_df[(training_df.IN_CR == 1) &
                                (training_df.IN_CS == 1) &
                                (training_df.IN_DS == 1)]\
                    .drop_duplicates(subset=OUT_OPTIONS)
                    [OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_balanced.shape[0])
display(distinct_balanced.head())


# In[8]:

distinct_total = training_df.drop_duplicates(subset=OUT_OPTIONS)
                            [OUT_OPTIONS + TEST_FEATURES]
print('%d distinct options from a total of %d' %
      (distinct_total.shape[0], 1620))
distinct_total_noblock = distinct_total.drop_duplicates(subset=OUT_OPTIONS[1:4])
print('%d distinct options from a total of %d' %
      (distinct_total_noblock.shape[0], 162))
print('Distinct codecs %d' % 
      distinct_total.drop_duplicates(subset=['Codec']).shape[0])
print('Distinct filters %d' %
      distinct_total.drop_duplicates(subset=['Filter']).shape[0])
print('Distinct CL %d' %
      distinct_total.drop_duplicates(subset=['CL']).shape[0])
print('Distinct block sizes %d' %
      distinct_total.drop_duplicates(subset=['Block_Size']).shape[0])
display(distinct_total.describe())


# Zlib queda descartado dado que nunca es seleccionado como óptimo.

# In[9]:

display(training_df[training_df.Codec == 'snappy']
        [IN_USER + TEST_FEATURES + OUT_OPTIONS])


# Snappy ha sido seleccionado en dos ocasiones. Por tanto podríamos considerar que tenemos 488/1080 opciones totales y sin contar el tamaño de bloque 97/108.

# In[10]:

print('%d blosclz classes from 270' %
      distinct_total[distinct_total.Codec == 'blosclz'].shape[0])
print('%d lz4 classes from 270' %
      distinct_total[distinct_total.Codec == 'lz4'].shape[0])
print('%d lz4hc classes from 270' %
      distinct_total[distinct_total.Codec == 'lz4hc'].shape[0])
print('%d zstd classes from 270' %
      distinct_total[distinct_total.Codec == 'zstd'].shape[0])


# Debido a que Snappy solo es seleccionado en dos ocasiones lo consideraremos como datos atípicos y por tanto los sustituimos por la segunda mejor opción.

# In[11]:

# ELIMINAMOS SNAPPY
for i, row in (training_df[training_df.Codec == 'snappy']).iterrows():
    aux = df[(df.Filename == row['Filename']) &
             (df.DataSet == row['DataSet']) &
             (df.Table == row['Table']) & 
             (df.Chunk_Number == row['Chunk_Number']) &
             (df.Codec != 'snappy')]
    i_max_crate, i_max_c_speed, i_max_d_speed = aux['CRate'].idxmax(),
                                                aux['CSpeed'].idxmax(),                                                aux['DSpeed'].idxmax()
    max_crate, max_c_speed, max_d_speed = (aux.ix[i_max_crate]['CRate'],
                                           aux.ix[i_max_c_speed]['CSpeed'],
                                           aux.ix[i_max_d_speed]['DSpeed'])

    min_crate, min_c_speed, min_d_speed = (aux['CRate'].min(),
                                           aux['CSpeed'].min(),
                                           aux['DSpeed'].min())
    # NORMALIZED COLUMNS
    aux = aux.assign(N_CRate=(aux['CRate'] - min_crate) 
                     / (max_crate - min_crate),
                     N_CSpeed=(aux['CSpeed'] - min_c_speed) 
                     / (max_c_speed - min_c_speed),
                     N_DSpeed=(aux['DSpeed'] - min_d_speed) 
                     / (max_d_speed - min_d_speed))
    aux['Distance'] = row['IN_CR']*(aux['N_CRate'] - 1)**2
                      + row['IN_DS']*(aux['N_DSpeed'] - 1)**2 
                      + row['IN_CS']*(aux['N_CSpeed'] - 1)**2
    index = aux['Distance'].idxmin()
    training_df.loc[i, TEST_FEATURES + OUT_OPTIONS] =         aux.ix[index][TEST_FEATURES + OUT_OPTIONS]


# ## Tamaño de bloque automático

# In[12]:

get_ipython().run_cell_magic('time', '', 'count = training_df[training_df.Block_Size == 0].shape[0]\nfor i, row in training_df.iterrows():\n    block = row[\'Block_Size\']\n    aux = df[(df.Filename == row[\'Filename\']) &\n             (df.DataSet == row[\'DataSet\']) &\n             (df.Table == row[\'Table\']) &\n             (df.Chunk_Number == row[\'Chunk_Number\']) &\n             (df.Codec == row[\'Codec\']) &\n             (df.Filter == row[\'Filter\']) &\n             (df.CL == row["CL"])]\n    crate = aux[aux.Block_Size == 0][\'CRate\'].values[0]\n    auto_block = aux[(aux.CRate == crate) &\n                     (aux.Block_Size != 0)][\'Block_Size\'].values[0]\n    if block != 0:\n        if auto_block == block:\n            count += 1\n    else:\n        training_df.loc[i, \'Block_Size\'] = auto_block')


# In[13]:

print("%d from %d --> %d %%" %
      (count, training_df.shape[0], count / training_df.shape[0] * 100))


# In[14]:

training_df.drop_duplicates(subset=['Block_Size'])


# ## Preparación de inputs para scikit-learn

# In[15]:

from sklearn.preprocessing import binarize 
from sklearn.preprocessing import OneHotEncoder
training_df = training_df.assign(
               is_Table=binarize(training_df['Table'].values.reshape(-1,1), 0), 
               is_Columnar=binarize(training_df['Table'].values.reshape(-1,1), 1),
               is_Int=training_df['DType'].str.contains('int').astype(int),
               is_Float=training_df['DType'].str.contains('float').astype(int),
               is_String=(training_df['DType'].str.contains('S') |
                          training_df['DType'].str.contains('U')).astype(int))
import re
def aux_func(s):
    n = int(re.findall('\d+', s)[0])
    isNum = re.findall('int|float', s)
    if len(isNum) > 0:
        return n // 8
    else:
        return n
training_df['Type_Size'] = [aux_func(s) for s in training_df['DType']]


# ## Preparación de outputs para scikit-learn

# In[16]:

training_df = training_df.assign(
                Blosclz=(training_df['Codec'] == 'blosclz').astype(int),
                Lz4=(training_df['Codec'] == 'lz4').astype(int),
                Lz4hc=(training_df['Codec'] == 'lz4hc').astype(int),
                Snappy=(training_df['Codec'] == 'snappy').astype(int),
                Zstd=(training_df['Codec'] == 'zstd').astype(int),
                Noshuffle=(training_df['Filter'] == 'noshuffle').astype(int),
                Shuffle=(training_df['Filter'] == 'shuffle').astype(int),
                Bitshuffle=(training_df['Filter'] == 'bitshuffle').astype(int))
enc_cl = OneHotEncoder()
enc_cl.fit(training_df['CL'].values.reshape(-1, 1))
new_cls = enc_cl.transform(training_df['CL'].values.reshape(-1, 1)).toarray()
enc_block = OneHotEncoder()
enc_block.fit(training_df['Block_Size'].values.reshape(-1, 1))
new_blocks = enc_block.transform(training_df['Block_Size']
                                 .values.reshape(-1, 1)).toarray()
for i in range(9):
    cl_label = 'CL' + str(i+1)
    block_label = 'Block_' + str(2**(i+3))
    training_df[cl_label] = new_cls[:, i]
    training_df[block_label] = new_blocks[:, i]


# In[17]:

training_df.to_csv('../data/training_data.csv', sep='\t', index=False)


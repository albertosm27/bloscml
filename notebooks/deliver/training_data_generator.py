
# coding: utf-8

# # Generador de los datos de entrenamiento

# ## Objetivos del análisis
# * Extraer el data frame final con los datos preparados para entrenar algoritmos machine learning.

# ## Descripción de la muestra
# 
# El DataFrame en cuestión está formado por las características extraídas de un array de datos al comprimirlo y descomprimirlo mediante blosc. En cada fichero aparecen distintos conjuntos de datos los cuáles dividimos en fragmentos de 16 MegaBytes y sobre los cuales realizamos las pruebas de compresión y decompresión.  
# Cada fila se corresponde con los datos de realizar los test de compresión sobre un fragmento (*chunk*) de datos específico con un tamaño de bloque, codec, filtro y nivel de compresión determinados.
# 
# Variable | Descripción
# -------------  | -------------
# *Filename* | nombre del fichero del que proviene.
# *DataSet* | dentro del fichero el conjunto de datos del que proviene.
# *Table* | 0 si los datos vienen de un array, 1 si vienen de tablas y 2 para tablas columnares.
# *DType* | indica el tipo de los datos.
# *Chunk_Number* | número de fragmento dentro del conjunto de datos.
# *Chunk_Size* | tamaño del fragmento.
# *Mean* | la media.
# *Median* | la mediana.
# *Sd* | la desviación típica.
# *Skew* | el coeficiente de asimetría.
# *Kurt* | el coeficiente de apuntamiento.
# *Min* | el mínimo absoluto.
# *Max* | el máximo absoluto.
# *Q1* | el primer cuartil.
# *Q3* | el tercer cuartil.
# *N_Streaks* | número de rachas seguidas por encima o debajo de la mediana.
# *Block_Size* | el tamaño de bloque que utilizará Blosc para comprimir.
# *Codec* | el codec de blosc utilizado.
# *Filter* | el filtro de blosc utilizado.
# *CL* | el nivel de compresión utilizado.
# *CRate* | el ratio de compresión obtenido.
# *CSpeed* | la velocidad de compresión obtenida en GB/s.
# *DSpeed* | la velocidad de decompresión obtenida en GB/s.

# In[1]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas')


# In[2]:

import numpy as np
import pandas as pd
from IPython.display import display

pd.options.display.float_format = '{:,.3f}'.format


# In[3]:

CHUNK_ID = ["Filename", "DataSet", "Table", "Chunk_Number"]
#CHUNK_FEATURES = ["Table", "DType", "Chunk_Size", "Mean", "Median", "Sd", "Skew", "Kurt", "Min", "Max", "Q1", "Q3", "N_Streaks"]
CHUNK_FEATURES = ["Table", "DType", "Chunk_Size", "Mean", "Median", "Sd", "Skew", "Kurt", "Min", "Max", "Q1", "Q3"]
OUT_OPTIONS = ["Block_Size", "Codec", "Filter", "CL"]
TEST_FEATURES = ["CRate", "CSpeed", "DSpeed"]
COLS = ["Filename" , "DataSet", "Chunk_Number"] + CHUNK_FEATURES + OUT_OPTIONS + TEST_FEATURES
IN_TESTS = ['BLZ_CRate', 'BLZ_CSpeed', 'BLZ_DSpeed', 'LZ4_CRate', 'LZ4_CSpeed', 'LZ4_DSpeed']
IN_USER = ['IN_CR', 'IN_CS', 'IN_DS']


# In[4]:

df = pd.read_csv('../data/blosc_test_data.csv.gz', sep='\t')
my_df = df[(df.Filename != 'WRF_India-LSD1.h5') & (df.Filename != 'WRF_India-LSD2.h5') 
           & (df.Filename != 'WRF_India-LSD3.h5') & (df.CL != 0) & (df.CRate > 1.1)]


# In[5]:

# DATAFRAME WITH DISTINCT CHUNKS
chunks_df = my_df.drop_duplicates(subset=CHUNK_ID)
print("%d rows" % chunks_df.shape[0])
chunk_tests_list = []
# FOR EACH CHUNK
for index, row in chunks_df.iterrows():
    # DATAFRAME WITH CHUNK TESTS
    chunk_tests_list.append(my_df[(my_df.Filename == row["Filename"]) & (my_df.DataSet == row["DataSet"]) &
                        (my_df.Table == row["Table"]) & (my_df.Chunk_Number == row["Chunk_Number"])])


# In[6]:

training_df = pd.DataFrame()
for chunk_test in chunk_tests_list:
    # EXTRACT MAX MIN AND SOME AUX MAX INDICES
    i_max_crate, i_max_c_speed, i_max_d_speed = chunk_test['CRate'].idxmax(), chunk_test['CSpeed'].idxmax(),                                                chunk_test['DSpeed'].idxmax()
    max_crate, max_c_speed, max_d_speed = (chunk_test.ix[i_max_crate]['CRate'], chunk_test.ix[i_max_c_speed]['CSpeed'],
                                           chunk_test.ix[i_max_d_speed]['DSpeed'])

    min_crate, min_c_speed, min_d_speed = (chunk_test['CRate'].min(), chunk_test['CSpeed'].min(),
                                           chunk_test['DSpeed'].min())
    # NORMALIZED COLUMNS
    chunk_test = chunk_test.assign(N_CRate=(chunk_test['CRate'] - min_crate) / (max_crate - min_crate),
                                   N_CSpeed=(chunk_test['CSpeed'] - min_c_speed) / (max_c_speed - min_c_speed),
                                   N_DSpeed=(chunk_test['DSpeed'] - min_d_speed) / (max_d_speed - min_d_speed))
    # DISTANCE FUNC COLUMNS
    chunk_test = chunk_test.assign(Distance_1=(chunk_test['N_CRate'] - 1)**2 + (chunk_test['N_CSpeed'] - 1)**2,
                                   Distance_2=(chunk_test['N_CRate'] - 1) ** 2 + (chunk_test['N_DSpeed'] - 1) ** 2,
                                   Distance_3=(chunk_test['N_CRate'] - 1) ** 2 + (chunk_test['N_DSpeed'] - 1) ** 2 +
                                              (chunk_test['N_CSpeed'] - 1) ** 2,
                                   Distance_4=(chunk_test['N_CSpeed'] - 1) ** 2 + (chunk_test['N_DSpeed'] - 1) ** 2
                                   )
    # BALANCED INDICES
    i_balanced_c_speed, i_balanced_d_speed, i_balanced, i_balanced_speeds = (chunk_test['Distance_1'].idxmin(),
                                                                             chunk_test['Distance_2'].idxmin(),
                                                                             chunk_test['Distance_3'].idxmin(),
                                                                             chunk_test['Distance_4'].idxmin())
    indices = [i_max_d_speed, i_max_c_speed, i_balanced_speeds, i_max_crate, i_balanced_d_speed, i_balanced_c_speed,
               i_balanced]
    # TYPE FILTER FOR LZ_DATA
    d_type = chunk_test.iloc[0]['DType']
    filter_name = 'noshuffle'
    if 'float' in d_type or 'int' in d_type:
        filter_name = 'shuffle'
    aux = df[(df.CL == 1) & (df.Block_Size == 0) & (df.Filter == filter_name) &
             (df.Filename == chunk_test.iloc[0]['Filename']) & (df.DataSet == chunk_test.iloc[0]['DataSet']) &
             (df.Table == chunk_test.iloc[0]['Table']) & (df.Chunk_Number == chunk_test.iloc[0]['Chunk_Number'])]
    lz_data = np.append(aux[aux.Codec == 'blosclz'][TEST_FEATURES].values[0],
                        aux[aux.Codec == 'lz4'][TEST_FEATURES].values[0])
    # APPEND ROWS TO TRAINING DATA FRAME
    for i in range(len(indices)):
        in_1, r = divmod((i+1), 4)
        in_2, in_3 = divmod(r, 2)
        training_df = training_df.append(dict(zip(COLS + IN_TESTS + IN_USER,
                                                  np.append(np.append(chunk_test.ix[indices[i]][COLS].values,
                                                                      lz_data),
                                                            [in_1, in_2, in_3]))),
                                         ignore_index=True)


# ## Algunas comprobaciones

# In[7]:

print('DISTINCT MAX RATE')
distinct_max_rate = training_df[(training_df.IN_CR == 1) & (training_df.IN_CS == 0) & (training_df.IN_DS == 0)]                    .drop_duplicates(subset=OUT_OPTIONS)[OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_max_rate.shape[0])
display(distinct_max_rate)
print('DISTINCT MAX C.SPEED')
distinct_max_c_speed = training_df[(training_df.IN_CR == 0) & (training_df.IN_CS == 1) & (training_df.IN_DS == 0)]                       .drop_duplicates(subset=OUT_OPTIONS)[OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_max_c_speed.shape[0])
display(distinct_max_c_speed)
print('DISTINCT MAX D.SPEED')
distinct_max_d_speed = training_df[(training_df.IN_CR == 0) & (training_df.IN_CS == 0) & (training_df.IN_DS == 1)]                      .drop_duplicates(subset=OUT_OPTIONS)[OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_max_d_speed.shape[0])
display(distinct_max_d_speed)
print('DISTINCT BALANCED CSPEED')
distinct_balanced_c_speed = training_df[(training_df.IN_CR == 1) & (training_df.IN_CS == 1) & (training_df.IN_DS == 0)]                            .drop_duplicates(subset=OUT_OPTIONS)[OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_balanced_c_speed.shape[0])
display(distinct_balanced_c_speed)
print('DISTINCT BALANCED DSPEED')
distinct_balanced_d_speed = training_df[(training_df.IN_CR == 1) & (training_df.IN_CS == 0) & (training_df.IN_DS == 1)]                            .drop_duplicates(subset=OUT_OPTIONS)[OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_balanced_d_speed.shape[0])
display(distinct_balanced_d_speed)
print('DISTINCT BALANCED SPEED')
distinct_balanced_speed = training_df[(training_df.IN_CR == 0) & (training_df.IN_CS == 1) & (training_df.IN_DS == 1)]                          .drop_duplicates(subset=OUT_OPTIONS)[OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_balanced_speed.shape[0])
display(distinct_balanced_speed)
print('DISTINCT BALANCED')
distinct_balanced = training_df[(training_df.IN_CR == 1) & (training_df.IN_CS == 1) & (training_df.IN_DS == 1)]                    .drop_duplicates(subset=OUT_OPTIONS)[OUT_OPTIONS + TEST_FEATURES]
print('%s rows' % distinct_balanced.shape[0])
display(distinct_balanced)


# In[8]:

distinct_total = training_df.drop_duplicates(subset=OUT_OPTIONS)[OUT_OPTIONS + TEST_FEATURES]
print('%d distinct options from a total of %d' % (distinct_total.shape[0], 1620))
distinct_total_noblock = distinct_total.drop_duplicates(subset=OUT_OPTIONS[1:4])
print('%d distinct options from a total of %d' % (distinct_total_noblock.shape[0], 162))
print('Distinct codecs %d' % distinct_total.drop_duplicates(subset=['Codec']).shape[0])
print('Distinct filters %d' % distinct_total.drop_duplicates(subset=['Filter']).shape[0])
print('Distinct CL %d' % distinct_total.drop_duplicates(subset=['CL']).shape[0])
print('Distinct block sizes %d' % distinct_total.drop_duplicates(subset=['Block_Size']).shape[0])
display(distinct_total.describe())


# Zlib ha muerto.

# In[9]:

# IMPRIMIMOS A NUESTRO MARCIANO FAVORITO
display(distinct_total[distinct_total.Codec == 'snappy'])


# Snappy está moribundo. Por tanto podríamos considerar que tenemos 488/1080 opciones totales y sin contar el tamaño de bloque 97/108.

# In[10]:

print('%d blosclz classes from 270' % distinct_total[distinct_total.Codec == 'blosclz'].shape[0])
print('%d lz4 classes from 270' % distinct_total[distinct_total.Codec == 'lz4'].shape[0])
print('%d lz4hc classes from 270' % distinct_total[distinct_total.Codec == 'lz4hc'].shape[0])
print('%d zstd classes from 270' % distinct_total[distinct_total.Codec == 'zstd'].shape[0])


# In[11]:

training_df.to_csv('../data/training_data.csv', sep='\t', index=False)


# ## Comprobación del tamaño de bloque automático

# In[15]:

print("%d from %d" % (training_df[training_df.Block_Size == 0].shape[0], training_df.shape[0]))


# In[31]:

count = 0
report = []
for indices, row in training_df.iterrows():
    block = row['Block_Size']
    if block != 0:
        aux = df[(df.Filename == row['Filename']) & (df.DataSet == row['DataSet']) &
                 (df.Table == row['Table']) & (df.Chunk_Number == row['Chunk_Number']) &
                 (df.Codec == row['Codec']) & (df.Filter == row['Filter']) & (df.CL == row["CL"])]
        crate = aux[aux.Block_Size == 0]['CRate'].values[0]
        result = aux[(aux.CRate == crate) & (aux.Block_Size != 0)]
        auto_block = result['Block_Size'].values[0]
        if auto_block == block:
            count += 1
        else:
            report.append((block, auto_block))


# In[34]:

print("%d from %d" % (training_df[training_df.Block_Size == 0].shape[0] + count, training_df.shape[0]))


# In[38]:

for line in report:
    print("Winner --> %d | Predicted --> %d" % line)


# In[ ]:




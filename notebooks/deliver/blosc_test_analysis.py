
# coding: utf-8

# # Análisis de las pruebas realizadas con Blosc 

# ## Objetivos del análisis
# * Relacionar el tamaño de bloque con las medidas de compresión y decompresión.
# * Comprobar el comportamiento de los niveles de compresión sobre las pruebas.
# * Comparar los datos de compresión de tablas normales y columnares.
# * ¿Existe correlación entre blosclz o lz4 con nivel de compresión 1 y el resto de codecs?
# * **[Punto muerto]** ¿Existe correlación entre las características del chunk y las medidas de compresión y decompresión?

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
import pandas as pd

import custom_plots as cst

pd.options.display.float_format = '{:,.3f}'.format
matplotlib.rcParams.update({'font.size': 12})


# ## Descripción general
# Cargamos el csv entero, comprobamos que no faltan campos y mostramos un breve resumen.

# In[3]:

# LOAD WHOLE CSV
DF = pd.read_csv('../data/blosc_test_data_v2.csv.gz', sep='\t')
# SORT COLUMNS
DF = DF[cst.COLS]
# CHECK MISSING DATA
if not DF.isnull().any().any():
    print('No missing data')
else:
    print("Missing data")


# In[4]:

# SUMMARY OF THE DATAFRAME 
display(DF[cst.COLS[5:]].describe())


# Filtramos el csv para eliminar ficheros que utilizan técnicas de compresión con pérdidas.

# In[5]:

my_df = DF[(DF.Filename != 'WRF_India-LSD1.h5') & (DF.Filename != 'WRF_India-LSD2.h5') & (DF.Filename != 'WRF_India-LSD3.h5')]


# Veamos cuantos conjuntos de datos tiene el fichero.

# In[6]:

sets = my_df.drop_duplicates(subset=['DataSet', 'Table'])[cst.DESC_SET]
display(sets)
print('There are %d datasets' % (sets.shape[0]))


# Procedemos a mostrar un resumen de las características extraídas de cada conjunto de datos.

# In[7]:

for dataset in sets.drop_duplicates(subset=['DataSet'])['DataSet']:
        set_info = sets[sets.DataSet == dataset]
        print('SUMMARY')
        print(set_info)
        aux_set = my_df[my_df.DataSet == dataset].drop_duplicates(subset=['Chunk_Number'])
        if aux_set.shape[0] > 1:
            display(aux_set.describe()[cst.CHUNK_FEATURES])
        else:
            display(aux_set[cst.CHUNK_FEATURES])


# Para evitar que los diagramas de caja esten plagados de datos atípicos, procedemos a filtrar con el codec blosclz, filtro shuffle, nivel de compresión 5 y tamaño de bloque automático para buscar con detenimiento datos atípicos.

# In[8]:

df_outliers = my_df[(my_df.Block_Size == 0) & (my_df.CL == 5) & (my_df.Codec == 'blosclz') & (my_df.Filter == 'noshuffle')]
cst.paint_dtype_boxplots(df_outliers)


# Mostramos a continuación los datos atípicos

# In[9]:

for i in range(2):
    dfaux = df_outliers[df_outliers.DType.str.contains(cst.TYPES[i])]
    if dfaux.size > 0:
        cr_lim = cst.outlier_lim(dfaux['CRate'])
        cs_lim = cst.outlier_lim(dfaux['CSpeed'])
        ds_lim = cst.outlier_lim(dfaux['DSpeed'])
        result = dfaux[(dfaux.CRate < cr_lim[0]) | (dfaux.CRate > cr_lim[1]) |
                      (dfaux.CSpeed < cs_lim[0]) | (dfaux.CSpeed > cs_lim[1]) |
                      (dfaux.DSpeed < ds_lim[0]) | (dfaux.DSpeed > ds_lim[1])][cst.ALL_FEATURES]
        if result.size > 0:
            print('%d %s OUTLIERS' % (result.shape[0], cst.TYPES[i].upper()))
            display(result)


# No mostramos los datos atípicos de tipo string dado que no extraemos ninguna característica de chunk que podamos comentar, nos centraremos en ellos cuando busquemos correlaciones entre blosclz y el resto de codecs.  
# En cuanto a los datos atípicos observamos que la mayoría son series números idénticos o muy parecidos, siempre con un rango intercuartílico de 0.

# ## Correlaciones Block Size
# Aquí pretendemos observar la correlación entre el tamaño de bloque y las medidas de compresión, para ello filtramos los datos por tipo, codec, filtro, nivel de compresión y tamaño de bloque; y calculamos la media de su ratio de compresión y velocidades de compresión/decompresión.
# 

# In[10]:

cst.paint_all_block_cor(my_df, 'shuffle', c_level=5)


# In[11]:

cst.paint_all_block_cor(my_df, 'noshuffle')


# In[12]:

cst.paint_cl_comparison(my_df, 'shuffle', 'blosclz')


# In[13]:

cst.paint_cl_comparison(my_df, 'shuffle', 'lz4')


# ## Comparación de niveles de compresión
# Al igual que en el anterior caso hacemos los mismos gráficos pero observando el nivel de compresión.

# In[14]:

# BLOCK SIZE --> CL
cst.paint_all_block_cor(my_df, 'shuffle', block_size=256, cl_mode=True)


# In[15]:

cst.paint_all_block_cor(my_df, 'noshuffle', block_size=256, cl_mode=True)


# ## Tablas columnares VS Tablas normales
# En el caso de que los datos esten en forma de tabla, si la tabla contiene más de una columna se realizan dos pruebas de compresión, una guardando los datos como tabla normal, fila por fila y otra guardándolos columnarmente.

# In[16]:

df_col = my_df[my_df.Table == 2]
if df_col.size > 0:
    sets = df_col.drop_duplicates(subset=['DataSet'])
    for dataset in sets['DataSet']:
        dfaux = my_df[my_df.DataSet == dataset]
        normal_table = dfaux[dfaux.Table == 1][cst.TEST_FEATURES]
        normal_table.columns = ['N_CRate', 'N_CSpeed', 'N_DSpeed']
        col_table = dfaux[dfaux.Table == 2][cst.TEST_FEATURES]
        col_table.columns = ['COL_CRate', 'COL_CSpeed', 'COL_DSpeed']
        result = pd.concat([normal_table, col_table])
        result = result[['N_CRate', 'COL_CRate', 'N_CSpeed', 'COL_CSpeed','N_DSpeed', 'COL_DSpeed']]
        print(sets[sets.DataSet == dataset][cst.DESC_SET])
        display(result.describe())


# ## Correlaciones Blosclz-CL1 VS Otros
# Para poder visualizar todas estas correlaciones calculamos directamente el coeficiente de pearson y su p-valor asociado entre los datos de blosclz con nivel de compresión 1 y el resto.

# In[17]:

cst.paint_codec_pearson_corr(my_df, 'blosclz', 1)


# In[18]:

cst.paint_codec_pearson_corr(my_df, 'lz4', 1)


# In[19]:

dfaux = my_df[(my_df.Codec == 'lz4') & (my_df.Block_Size == 256) & (my_df.Filter == 'shuffle') &
              (my_df.CL == 5) & (my_df.DType.str.contains('float') | my_df.DType.str.contains('int'))]
cols = ['Mean', 'Sd', 'Skew', 'Kurt']
cst.custom_pairs(dfaux, cols)


# In[20]:

# TODO N_STREAKS
cols = ['Range', 'Q_Range']
dfaux = dfaux.assign(Range=dfaux['Max'] - dfaux['Min'])
dfaux = dfaux.assign(Q_Range=dfaux['Q3'] - dfaux['Q1'])
cst.custom_pairs(dfaux, cols)


# In[ ]:




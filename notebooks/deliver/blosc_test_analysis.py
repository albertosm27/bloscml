
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
my_df = pd.read_csv('../data/blosc_test_data_final.csv.gz', sep='\t')
# SORT COLUMNS
my_df = my_df[cst.COLS]
# CHECK MISSING DATA
if not my_df.isnull().any().any():
    print('No missing data')
else:
    print("Missing data")


# In[4]:

# SUMMARY OF THE DATAFRAME 
display(my_df[cst.COLS[5:]].describe())


# Como se puede observar hay mucha variabilidad en nuestros datos, lo cual es bueno.

# Veamos cuantos conjuntos de datos tiene el fichero.

# In[5]:

sets = my_df.drop_duplicates(subset=['DataSet', 'Table'])[cst.DESC_SET]
print('First ten datasets')
display(sets.head(n=10))
print('There are %d datasets' % (sets.shape[0]))


# ## Tablas de referencia de los conjuntos de datos
# Procedemos a mostrar un resumen de las características extraídas de cada conjunto de datos.

# In[6]:

for dataset in sets.drop_duplicates(subset=['DataSet'])['DataSet']:
        set_info = sets[sets.DataSet == dataset]
        print('SUMMARY')
        print(set_info)
        aux_set = my_df[my_df.DataSet == dataset].drop_duplicates(subset=['Chunk_Number'])
        if aux_set.shape[0] > 1:
            display(aux_set.describe()[cst.CHUNK_FEATURES])
        else:
            display(aux_set[cst.CHUNK_FEATURES])


# No entraremos en detalles sobre cada conjunto de datos, simplemente nos conviene tener estas tablas como referencia rápida en caso de detectar anomalías en algún conjunto en concreto.

# ## Detección d
# Para evitar que los diagramas de caja esten plagados de datos atípicos, procedemos a filtrar con el codec blosclz, filtro shuffle, nivel de compresión 5 y tamaño de bloque automático para buscar con detenimiento datos atípicos.

# In[7]:

df_outliers = my_df[(my_df.Block_Size == 0) & (my_df.CL == 5) &
                    (my_df.Codec == 'blosclz') &
                    (my_df.Filter == 'noshuffle')]
cst.paint_dtype_boxplots(df_outliers)


# Mostramos a continuación los datos atípicos

# In[20]:

for i in range(2):
    dfaux = df_outliers[df_outliers.DType.str.contains(cst.TYPES[i])]
    if dfaux.size > 0:
        cr_lim = cst.outlier_lim(dfaux['CRate'])
        cs_lim = cst.outlier_lim(dfaux['CSpeed'])
        ds_lim = cst.outlier_lim(dfaux['DSpeed'])
        result = dfaux[(dfaux.CRate < cr_lim[0]) | 
                       (dfaux.CRate > cr_lim[1]) |
                       (dfaux.CSpeed < cs_lim[0]) | 
                       (dfaux.CSpeed > cs_lim[1]) |
                       (dfaux.DSpeed < ds_lim[0]) | 
                       (dfaux.DSpeed > ds_lim[1])][cst.ALL_FEATURES]
        if result.size > 0:
            print('%d %s OUTLIERS' % (result.shape[0],
                                      cst.TYPES[i].upper()))
            display(result.head())


# No mostramos los datos atípicos de tipo string dado que no extraemos ninguna característica de chunk que podamos comentar.  
# En cuanto a los datos atípicos observamos que la mayoría son series de números idénticos o muy parecidos, siempre con un rango intercuartílico de 0.

# ## Correlaciones Block Size
# Aquí pretendemos observar la correlación entre el tamaño de bloque y las medidas de compresión, para ello filtramos los datos por tipo, codec, filtro, nivel de compresión y tamaño de bloque; y calculamos la media de su ratio de compresión y velocidades de compresión/decompresión.
# 
# Las gráficas presentan los ratios de compresión (en azul) y las velocidades de compresión y de descompresión (en rojo y verde) medios para cada tamaño de bloque. Primero mostramos estos datos para los datos de tipo float y de tipo int.

# In[9]:

cst.paint_all_block_cor(my_df, 'shuffle', c_level=5)


# Aquí se muestran los mismos gráficos pero para los datos del tipo cadenas de texto

# In[10]:

cst.paint_all_block_cor(my_df, 'noshuffle')


# Como podemos observar, al aumentar el tamaño de bloque suele aumentar el ratio de compresión pero parece converger hasta un límite entre los tamaños de 512 KB y 2 MB. Además cuando el tamaño de bloque es menor en general las velocidades son más rápidas.
# 
# Por otro lado destaca el comportamiento de Snappy pues no parece comprimir muy bien con respecto al resto. Por otro lado Zlib parece ser inferior en todo a Zstd.

# Aquí se presentan las mismas gráficas pero alterando el nivel de compresión para ver como afecta al tamaño de bloque.

# In[11]:

cst.paint_cl_comparison(my_df, 'shuffle', 'blosclz')


# In[12]:

cst.paint_cl_comparison(my_df, 'shuffle', 'lz4')


# Los resultados son los esperados el comportamiento es en general el mismo, simplemente suben los ratio de compresión y bajan las velocidades a medida que aumenta el nivel de compresión. Por otra parte destaca el comportamiento del tamaño de bloque automático observamos que está programado para que aumente conjuntamente con el nivel de compresión.

# ## Comparación de niveles de compresión
# Al igual que en el anterior caso hacemos los mismos gráficos pero observando el nivel de compresión.

# In[13]:

# BLOCK SIZE --> CL
cst.paint_all_block_cor(my_df, 'shuffle', block_size=256, cl_mode=True)


# In[14]:

cst.paint_all_block_cor(my_df, 'noshuffle', block_size=256, cl_mode=True)


# Destaca el comportamiento de Snappy de nuevo vuelve a ser el más raro de todos, el nivel de compresión no cambia nada. Por otro lado Zlib tiene un cambio brusco a partir del nivel de compresión 3, esto se debe a que a partir de ese nivel activa métodos más potentes a la hora de comprimir. Finalmente Zstd parece hacer lo mismo que Zlib, pero parece que en los últimos niveles de compresión no funciona bien, pues pierde ratio de compresión.

# ## Tablas columnares VS Tablas normales
# En el caso de que los datos esten en forma de tabla, si la tabla contiene más de una columna se realizan dos pruebas de compresión, una guardando los datos como tabla normal, fila por fila y otra guardándolos columnarmente.

# In[15]:

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
        result = result[['N_CRate', 'COL_CRate', 'N_CSpeed', 
                         'COL_CSpeed','N_DSpeed', 'COL_DSpeed']]
        print(sets[sets.DataSet == dataset][cst.DESC_SET])
        display(result.describe())


# Como era de esperar, parece que las tablas columnares son más comprimibles. Aunque hay casos en los que se comprimen igual, nunca se comprimen menos.

# ## Correlaciones Blosclz-CL1 VS Otros
# Para poder visualizar todas estas correlaciones calculamos directamente el coeficiente de pearson asociado entre los datos de blosclz con nivel de compresión 1 y el resto.

# In[16]:

cst.paint_codec_pearson_corr(my_df, 'blosclz', 1)


# Aquí hacemos los mismo para LZ4

# In[17]:

cst.paint_codec_pearson_corr(my_df, 'lz4', 1)


# Los resultados son bastante buenos, además era de esperar. Aunque LZ4 tiene mejores resultados ambas opciones parecen lo suficientemente buenas.

# ## Correlaciónes entre características de chunk y pruebas de compresión
# Aquí se trata de observar las correlaciones entre características de chunk seleccionadas y las pruebas de compresiones. Para ello se utiliza un gráfico de pares personalizado. Además los datos se filtran por codec, filtro, nivel de compresión y tamaño de bloque, sino no tendría sentido los gráficos debido a la enorme variabilidad que habría.

# In[18]:

dfaux = my_df[(my_df.Codec == 'lz4') & (my_df.Block_Size == 256) &
              (my_df.Filter == 'shuffle') & (my_df.CL == 5) &
              (my_df.DType.str.contains('float') |
               my_df.DType.str.contains('int'))]
cols = ['Mean', 'Sd', 'Skew', 'Kurt']
cst.custom_pairs(dfaux, cols)


# In[19]:

cols = ['Range', 'Q_Range', 'N_Streaks']
dfaux = dfaux.assign(Range=dfaux['Max'] - dfaux['Min'])
dfaux = dfaux.assign(Q_Range=dfaux['Q3'] - dfaux['Q1'])
cst.custom_pairs(dfaux, cols)


# Aunque se podría plantear decir que a mayor rango y número de rachas disminuye el ratio de compresión, no sería muy adecuado sacar conclusiones de estos gráficos. Hay demasiada variabilidad en los datos en sí como para extraer conclusiones de un simple gráfico, será mejor que estas correlaciones las busquen los algoritmos de clasificación en sí.

from __future__ import print_function
import numpy as np
import blosc
import tables
import functools
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
from graphic_features import test_codec

MEGA32 = int((1*2**20)/4)

def chunk_32_generator(buffer):
    """
        chunk_32_generator(buffer)

        Given a buffer array of 32 bits elements, generates data chunks of 1MB.

        Parameters
        ----------
        buffer : a buffer array of 32 bits elements.

        Returns
        -------
        out : array
            A part of 1MB of extracted from the original buffer.
    """
    max,r = divmod(buffer.size, MEGA32)
    for i in range(max):
        yield buffer[i * MEGA32 : (i + 1) * MEGA32]
    if (r != 0):
        yield buffer[max * MEGA32: buffer.size]

def outlier_detection(array):
    """
        outlier_detection(array)

        Given an array of data detects outliers based on the Tukey's test. A data
        is considered an outlier when falls outside of the range [Q1-1.5(Q3-Q1), Q3+1.5(Q3-Q1)]
        where Q1 and Q3 are the lower and upper quartiles.

        Parameters
        ----------
        array : the array of data.

        Returns
        -------
        out : list
            A list with the indices of the data considered outliers.
    """
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    min = q1 - 1.5 * (q3 - q1)
    max = q3 + 1.5 * (q3 - q1)
    indices = []
    for i in range(array.size):
        if (array[i] < min or array[i] > max):
            indices.append(i)
    return indices

C_LEVELS = (1,9)
CODECS = ('blosclz', 'lz4')
FILE_NAME = 'F:DADES/WRF_India-LSD1.h5'

f = tables.open_file(FILE_NAME)
buffer = f.root.U[:].reshape(functools.reduce(lambda x, y: x*y, f.root.U.shape))
# for child in f.root._f_walknodes():
#     child[:].reshape(child.size)
f.close()
q, r = divmod(buffer.size, MEGA32)
N_CHUNKS = q
if (r != 0):
    N_CHUNKS += 1

tblz = np.empty(N_CHUNKS, dtype=float)
rblz = np.empty(N_CHUNKS, dtype=float)
tblz9 = np.empty(N_CHUNKS, dtype=float)
rblz9 = np.empty(N_CHUNKS, dtype=float)
tzstd = np.empty(N_CHUNKS, dtype=float)
rzstd = np.empty(N_CHUNKS, dtype=float)
tlz4 = np.empty(N_CHUNKS, dtype=float)
rlz4 = np.empty(N_CHUNKS, dtype=float)
tlz49 = np.empty(N_CHUNKS, dtype=float)
rlz49 = np.empty(N_CHUNKS, dtype=float)
tlz4hc = np.empty(N_CHUNKS, dtype=float)
rlz4hc = np.empty(N_CHUNKS, dtype=float)
tsnappy = np.empty(N_CHUNKS, dtype=float)
rsnappy = np.empty(N_CHUNKS, dtype=float)
tzlib = np.empty(N_CHUNKS, dtype=float)
rzlib = np.empty(N_CHUNKS, dtype=float)

for i, chunk in enumerate(chunk_32_generator(buffer)):
    tblz[i], rblz[i] = test_codec(chunk, 'blosclz', blosc.SHUFFLE, 1)
    tblz9[i], rblz9[i] = test_codec(chunk, 'blosclz', blosc.SHUFFLE, 9)
    tzstd[i], rzstd[i] = test_codec(chunk, 'zstd', blosc.SHUFFLE, 1)
    tlz4[i], rlz4[i] = test_codec(chunk, 'lz4', blosc.SHUFFLE, 1)
    tlz49[i], rlz49[i] = test_codec(chunk, 'lz4', blosc.SHUFFLE, 9)
    tlz4hc[i], rlz4hc[i] = test_codec(chunk, 'lz4hc', blosc.SHUFFLE, 1)
    tsnappy[i], rsnappy[i] = test_codec(chunk, 'snappy', blosc.SHUFFLE, 1)
    tzlib[i], rzlib[i] = test_codec(chunk, 'zlib', blosc.SHUFFLE, 1)

# Outliers removal
def outlier_removal(indices):
    print('Removing outliers at: ', indices)
    np.delete(tblz, indices)
    np.delete(tblz9, indices)
    np.delete(tzstd, indices)
    np.delete(tlz4, indices)
    np.delete(tlz49, indices)
    np.delete(tlz4hc, indices)
    np.delete(tsnappy, indices)
    np.delete(tzlib, indices)

outlier_removal(outlier_detection(tblz))
outlier_removal(outlier_detection(tblz9))
outlier_removal(outlier_detection(tzstd))
outlier_removal(outlier_detection(tlz4))
outlier_removal(outlier_detection(tlz49))
outlier_removal(outlier_detection(tlz4hc))
outlier_removal(outlier_detection(tsnappy))
outlier_removal(outlier_detection(tzlib))

for codec in CODECS:
    for c_level in C_LEVELS:
        print('CORRELATIONS WITH ', codec.upper(), ' AND COMPRESSION LEVEL ', c_level)
        # PEARSON R
        if (codec == 'lz4'):
            codec_aux = 'blosclz'
            raux = rblz
            taux = tblz
            if (c_level == 1):
                rates = rlz4
                times = tlz4
            else:
                rates = rlz49
                times = tlz49
        else:
            codec_aux = 'lz4'
            raux = rlz4
            taux = tlz4
            if (c_level == 1):
                rates = rblz
                times = tblz
            else:
                rates = rblz9
                times = tblz9
        print("-------COMPRESSION RATES---------")
        print("Pearson ", codec, "- zstd: ", pearsonr(rates, rzstd))
        print("Pearson ", codec, "-", codec_aux, ": ", pearsonr(rates, raux))
        print("Pearson ", codec, "- lz4hc: ", pearsonr(rates, rlz4hc))
        print("Pearson ", codec, "- snappy: ", pearsonr(rates, rsnappy))
        print("Pearson ", codec, "- zlib: ", pearsonr(rates, rzlib))
        print("-------COMPRESSION TIMES---------")
        print("Pearson ", codec, "- zstd: ", pearsonr(times, tzstd))
        print("Pearson ", codec, "-", codec_aux, ": ", pearsonr(times, taux))
        print("Pearson ", codec, "- lz4hc: ", pearsonr(times, tlz4hc))
        print("Pearson ", codec, "- snappy: ", pearsonr(times, tsnappy))
        print("Pearson ", codec, "- zlib: ", pearsonr(times, tzlib), '\n')

# 2D GRAPHICS
f, axarr = plt.subplots(3,2)
axarr[0,0].scatter(rblz, rzstd, c='red')
axarr[0,0].set_title('RATES: blosclz VS zstd')

axarr[0,1].scatter(tblz, tzstd, c='red')
axarr[0,1].set_title('TIMES: blosclz VS zstd')

axarr[1,0].scatter(rblz, rlz4, c='red')
axarr[1,0].set_title('RATES: blosclz VS lz4')

axarr[1,1].scatter(tblz, tlz4, c='red')
axarr[1,1].set_title('TIMES: blosclz VS lz4')

axarr[2,0].scatter(rblz, rlz4hc, c='red')
axarr[2,0].set_title('RATES: blosclz VS lz4hc')

axarr[2,1].scatter(tblz, tlz4hc, c='red')
axarr[2,1].set_title('TIMES: blosclz VS lz4hc')

f2, axarr2 = plt.subplots(2,2)
axarr2[0,0].scatter(rblz, rsnappy, c='red')
axarr2[0,0].set_title('RATES: blosclz VS snappy')

axarr2[0,1].scatter(tblz, tsnappy, c='red')
axarr2[0,1].set_title('TIMES: blosclz VS snappy')

axarr2[1,0].scatter(rblz, rzlib, c='red')
axarr2[1,0].set_title('RATES: blosclz VS zlib')

axarr2[1,1].scatter(tblz, tzlib, c='red')
axarr2[1,1].set_title('TIMES: blosclz VS zlib')

plt.show()

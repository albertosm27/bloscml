from __future__ import print_function
import blosc
import tables
import functools
import numpy as np
import scipy.stats as stats
import pandas as pd
import os.path
from sys import platform
if platform == 'win32':
    from time import clock as time
else:
    from time import time as time

def test_codec( chunk, codec, filter, clevel ):
    """
    Compresses the array chunk with the given codec, filter and clevel
    and return the compression time and rate.

    Parameters
    ----------
    chunk : bytes-like object (supporting the buffer interface)
        The data to be compressed.
    codec : string
        The name of the compressor used internally in Blosc. It can be
        any of the supported by Blosc ('blosclz', 'lz4', 'lz4hc',
        'snappy', 'zlib', 'zstd' and maybe others too).
    clevel : int
        The compression level from 0 (no compression) to 9
        (maximum compression).
    shuffle : int
        The shuffle filter to be activated.  Allowed values are
        blosc.NOSHUFFLE, blosc.SHUFFLE and blosc.BITSHUFFLE.

    Returns
    -------
    out : tuple
        The associated compression time, rate and decompression time.

    Raises
    ------
    TypeError
        If bytesobj doesn't support the buffer interface.
    ValueError
        If bytesobj is too long.
        If typesize is not within the allowed range.
        If clevel is not within the allowed range.
        If cname is not a valid codec.
    """
    t0 = time()
    c = blosc.compress_ptr(chunk.__array_interface__['data'][0],
                           chunk.size, chunk.dtype.itemsize,
                           clevel = clevel, shuffle = filter, cname = codec)
    tc = time() - t0
    out = np.empty(chunk.size, dtype = chunk.dtype)
    t0 = time()
    blosc.decompress_ptr(c, out.__array_interface__['data'][0])
    td = time() - t0
    rate = (chunk.size * chunk.dtype.itemsize / len(c))
    assert ((chunk == out).all())
    # print("  *** %-8s, %-10s, CL%d *** %6.4f s / %5.4f s " %
    #        ( codec, blosc.filters[filter], clevel, tc, td), end='')
    # print("\tCompr. ratio: %5.1fx" % rate)
    return (rate, tc, td)

def mega_chunk_generator(buffer):
    """
    Given a buffer array generates data chunks of 1MB.

    Parameters
    ----------
    buffer : a buffer array.

    Returns
    -------
    out : array
        A part of 1MB of extracted from the original buffer.
    """
    mega = int((2**20) / buffer.dtype.itemsize)
    max, r = divmod(buffer.size, mega)
    for i in range(max):
        yield buffer[i * mega: (i + 1) * mega]
    if (r != 0):
        yield buffer[max * mega: buffer.size]

def file_reader(filename):
    """
    Given an HDF5 file generates the buffers of data contained
    in the file.

    Parameters
    ----------
    filename : the name of the HDF5 file.

    Returns
    -------
    out : array
        A buffer of data contained in the file.
    """
    with tables.open_file(filename) as f:
        for child in f.root._f_walknodes():
            yield child[:].reshape(functools.reduce(lambda x, y: x*y, child.shape))

def col_labels():
    """
    Generates the array with the column labels.
    """
    for codec in blosc.compressor_list():
        for filter in [blosc.NOSHUFFLE, blosc.SHUFFLE, blosc.BITSHUFFLE]:
            for clevel in range(10):
                col_label = codec + str(filter) + str(clevel)
                COLS.append(col_label + '_r')
                COLS.append(col_label + '_tc')
                COLS.append(col_label + '_td')

def extract_features(buffer):
    """
    Extracts the mean, median, standard deviations from the buffer.

    Parameters
    ----------
    buffer : a buffer array of numbers.

    Returns
    -------
    out : tuple
        A tuple containing the mean, median and standard deviation.
    """
    return (np.mean(buffer), np.median(buffer), np.std(buffer), stats.skew(buffer), stats.kurtosis(buffer))

FILENAMES = ['WRF_India-LSD1.h5']
COLS = ['DataID', 'mean', 'median', 'sd', 'skew', 'kurt']
col_labels()

if os.path.isfile('out.csv'):
    df = pd.read_csv('out.csv', sep='\t')
else:
    df = pd.DataFrame()

for filename in FILENAMES:
    for k, buffer in enumerate(file_reader(filename)):
        row_data = [filename.split('/')[-1] + '_' + str(k)]
        for data in extract_features(buffer):
            row_data.append(data)
        aux = 1
        print("------------", filename.upper(), k, "------------")
        for codec in blosc.compressor_list():
            for filter in [blosc.NOSHUFFLE, blosc.SHUFFLE, blosc.BITSHUFFLE]:
                for clevel in range(10):
                    rate = 0
                    c_time = 0
                    d_time = 0
                    for i, chunk in enumerate(mega_chunk_generator(buffer)):
                        test = test_codec(chunk, codec, filter, clevel)
                        rate = (rate * i + test[0]) / (i + 1)
                        c_time = (c_time * i + test[1]) / (i + 1)
                        d_time = (d_time * i + test[2]) / (i + 1)
                    print("%-10s  %5.2f %%" % (codec + str(filter) + str(clevel), aux/180*100))
                    aux += 1
                    row_data.append(rate)
                    row_data.append(c_time)
                    row_data.append(d_time)
        df = df.append(dict(zip(COLS, row_data)), ignore_index=True)
        print('\n\nROW ADDED\n', df)
        df.to_csv('out.csv', sep='\t', index=False)
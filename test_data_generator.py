from __future__ import print_function
import blosc
import tables
import functools
import numpy as np
import scipy.stats as stats
import pandas as pd
import os.path
from queue import Queue
from sys import platform
if platform == 'win32':
    from time import clock as time
else:
    from time import time as time


SPEED_UNIT = 2**30
CHUNK_SIZE = 2**24
MINIMUM_SIZE = 2**13
KB32, KB128, MB = 2**15, 2**17, 2**20
KB64, KB256, MB2 = 2**16, 2**18, 2**21
KB16, KB512 = 2**14, 2**19


def test_codec(chunk, codec, filter_name, clevel):
    """
    Compress the chunk and return tested data.

    Parameters
    ----------
    chunk: bytes-like object (supporting the buffer interface)
        The data to be compressed.
    codec : string
        The name of the compressor used internally in Blosc. It can be
        any of the supported by Blosc ('blosclz', 'lz4', 'lz4hc',
        'snappy', 'zlib', 'zstd' and maybe others too).
    filter_name : int
        The shuffle filter to be activated.  Allowed values are
        blosc.NOSHUFFLE, blosc.SHUFFLE and blosc.BITSHUFFLE.
    clevel : int
        The compression level from 0 (no compression) to 9
        (maximum compression).
    Returns
    -------
    out: tuple
        The associated compression rate, compression speed and
        decompression speed (in GB/s).
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
                           clevel=clevel, shuffle=filter_name, cname=codec)
    tc = time() - t0
    out = np.empty(chunk.size, dtype=chunk.dtype)
    times = []
    for i in range(3):
        t0 = time()
        blosc.decompress_ptr(c, out.__array_interface__['data'][0])
        times.append(time() - t0)
    chunk_byte_size = chunk.size * chunk.dtype.itemsize
    rate = chunk_byte_size / len(c)
    c_speed = chunk_byte_size / tc / SPEED_UNIT
    d_speed = chunk_byte_size / min(times) / SPEED_UNIT
    # print("  *** %-8s, %-10s, CL%d *** %6.4f s / %5.4f s " %
    #        ( codec, blosc.filters[filter], clevel, tc, td), end='')
    # print("\tCompr. ratio: %5.1fx" % rate)
    return rate, c_speed, d_speed


def chunk_generator(buffer):
    """
    Generate data chunks of 16 MB in `buffer`.

    Parameters
    ----------
    buffer : numpy.array
        Buffer array of data

    Returns
    -------
    out : numpy.array
        A chunk of 16 MB extracted from the original buffer.
    """
    mega = int(CHUNK_SIZE / buffer.dtype.itemsize)
    max, r = divmod(buffer.size, mega)
    for i in range(max):
        yield buffer[i * mega: (i + 1) * mega]
    if r != 0:
        yield buffer[max * mega: buffer.size]


def file_reader(filename):
    """
    Generate the buffers of data in `filename`.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file.

    Returns
    -------
    out : tuple
        A tuple with the path to the data, the dtype, a number 0 (not a table), 1 (table) or 2 (column wise table)
        and the numpy array with the data.
    """
    with tables.open_file(filename) as f:
        group_queue = Queue()
        group_queue.put(f.root)
        while not group_queue.empty():
            try:
                for child in group_queue.get():
                    if hasattr(child, 'dtype'):
                        if child.size_in_memory > MINIMUM_SIZE:
                            if hasattr(child, 'colnames'):
                                for col_name in child.colnames:
                                    col = child.col(col_name)
                                    if col.size * col.dtype.itemsize > MINIMUM_SIZE:
                                        yield child._v_pathname + '.' + col_name, col.dtype, 1, \
                                            col[:].reshape(functools.reduce(
                                                lambda x, y: x * y, col.shape))
                                        col_shape = child.description.__getattribute__(
                                            col_name).shape
                                        if len(col_shape) > 1 or (len(col_shape) > 0 and col_shape[0] > 1):
                                            yield child._v_pathname + '.' + col_name, col.dtype, 2, \
                                                np.moveaxis(col, 0, -1)[:]\
                                                .reshape(functools.reduce(lambda x, y: x * y, col.shape))
                            else:
                                yield child._v_pathname, child.dtype, 0, \
                                    child[:].reshape(functools.reduce(
                                        lambda x, y: x * y, child.shape))
                    elif hasattr(child, '_v_children'):
                        group_queue.put(child)
            except TypeError:
                pass
            except tables.HDF5ExtError:
                pass


def calculate_nchunks(type_size, buffer_size):
    """
    Calculate the number of chunks.

    Parameters
    ----------
    type_size : int
        The type size in bytes.
    buffer_size : int
        The buffer size in number of elements.

    Returns
    -------
    out : int
        The number of chunks associated with the buffer and chunk size.
    """
    chunks_aux = int(CHUNK_SIZE / type_size)
    q, r = divmod(buffer_size, chunks_aux)
    n_chunks = q
    if r != 0:
        n_chunks += 1
    return n_chunks


def extract_chunk_features(chunk):
    """
    Extract the statistics features in `chunk`.

    Parameters
    ----------
    chunk : numpy.array
        An array of numbers.

    Returns
    -------
    out : tuple
        A tuple containing the mean, median, standard deviation, skewness,
        kurtosis, minimum, maximum and quartiles.
    """
    if np.isnan(chunk).any():
        return np.nanmean(chunk), np.nanmedian(chunk), np.nanstd(chunk), stats.skew(chunk, nan_policy='omit'),\
            stats.kurtosis(chunk, nan_policy='omit'), np.nanmin(chunk), np.nanmax(chunk),\
            np.nanpercentile(chunk, 25), np.nanpercentile(chunk, 75)
    else:
        return np.mean(chunk), np.median(chunk), np.std(chunk), stats.skew(chunk), stats.kurtosis(chunk),\
            np.min(chunk), np.max(chunk), np.percentile(
                chunk, 25), np.percentile(chunk, 75)


def calculate_streaks(chunk, median):
    """
    Calculate number of streaks.

    Parameters
    ----------
    chunk : numpy.array
        An array of numbers.
    median : number
        The median of the chunk.

    Returns
    -------
    out : int
        Number of streaks above/below median of the chunk.
    """
    streaks = 1
    above = chunk[0] > median
    for number in chunk[1:]:
        if above != (number > median):
            streaks += 1
            above = not above
    return streaks


FILENAMES = ('HiSPARC.h5',)
PATH = '/home/francesc/datasets/tests/'
BLOCK_SIZES = (0, MINIMUM_SIZE, KB16, KB32, KB64, KB128, KB256, KB512, MB, MB2)
C_LEVELS = range(1, 10)
COLS = ['Filename', 'DataSet', 'Table', 'DType', 'Chunk_Number', 'Chunk_Size', 'Mean', 'Median', 'Sd', 'Skew', 'Kurt',
        'Min', 'Max', 'Q1', 'Q3', 'N_Streaks', 'Block_Size', 'Codec', 'Filter', 'CL', 'CRate', 'CSpeed', 'DSpeed']
blosc.set_nthreads(4)

if not os.path.isfile('blosc_test_data.csv'):
    pd.DataFrame(columns=COLS).to_csv(
        'blosc_test_data.csv', sep='\t', index=False)

for filename in FILENAMES:
    for path, d_type, table, buffer in file_reader(PATH + filename):
        n_chunks = calculate_nchunks(buffer.dtype.itemsize, buffer.size)
        print("Starting tests with %-s %-s t%-s" % (filename, path, table))
        if buffer.dtype.kind in ('S', 'U'):
            is_string = True
            filters = (blosc.NOSHUFFLE,)
        else:
            is_string = False
            filters = (blosc.NOSHUFFLE, blosc.SHUFFLE, blosc.BITSHUFFLE)
        for i, chunk in enumerate(chunk_generator(buffer)):
            if is_string:
                chunk_features = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            else:
                chunk_features = extract_chunk_features(chunk)
                chunk_features += (calculate_streaks(chunk,
                                                     chunk_features[1]),)
            df = pd.DataFrame()
            for block_size in BLOCK_SIZES:
                blosc.set_blocksize(block_size)
                for codec in blosc.compressor_list():
                    for filter in filters:
                        for clevel in C_LEVELS:
                            row_data = (filename, path, table, d_type, i + 1,
                                        chunk.size * chunk.dtype.itemsize / MB) \
                                + chunk_features \
                                + (block_size / 2**10, codec, blosc.filters[filter], clevel) \
                                + test_codec(chunk, codec,
                                             filter, clevel)
                            df = df.append(
                                dict(zip(COLS, row_data)), ignore_index=True)
            print("%5.2f%% %-s %-s t%-s chunk %d completed" %
                  ((i + 1) / n_chunks * 100, filename, path, table, (i + 1)))
            with open('blosc_test_data.csv', 'a') as f:
                df = df[COLS]
                df.to_csv(f, sep='\t', index=False, header=False)
            print('CHUNK WRITED')

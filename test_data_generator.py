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

def test_codec(chunk, codec, filter, clevel):
    """
    Compresses the array chunk with the given codec, filter and clevel
    and return the compressions speeds and rate.

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
        The associated compression rate, compression speed and
        decompression speed (speed in GB/s).

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
    chunk_byte_size = chunk.size * chunk.dtype.itemsize
    rate = (chunk_byte_size / len(c))
    assert ((chunk == out).all())
    c_speed = (chunk_byte_size / tc / 2**30)
    d_speed = (chunk_byte_size / td / 2**30)
    # print("  *** %-8s, %-10s, CL%d *** %6.4f s / %5.4f s " %
    #        ( codec, blosc.filters[filter], clevel, tc, td), end='')
    # print("\tCompr. ratio: %5.1fx" % rate)
    return (rate, c_speed, d_speed)

def chunk_generator(buffer):
    """
    Given a buffer array generates data chunks of 16MB.

    Parameters
    ----------
    buffer : array
        Buffer array of data

    Returns
    -------
    out : array
        A part of 16MB extracted from the original buffer.
    """
    mega = int(2**24  / buffer.dtype.itemsize)
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
    filename : str
        The name of the HDF5 file.

    Returns
    -------
    out : array
        A buffer of data contained in the file.
    """
    # TODO IMPROVE DEEP BEHAVIOUR
    with tables.open_file(filename) as f:
        for child in f.root:
            if (child.size_in_memory > 10 * 2**20):
                yield (child._v_pathname, child[:].reshape(functools.reduce(lambda x, y: x * y, child.shape)))


def calculate_nchunks(type_size, buffer_size):
    """
    Calculates the number of chunks for the buffer.

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
    chunks_aux = int(2**24 / type_size)
    q, r = divmod(buffer_size, chunks_aux)
    n_chunks = q
    if (r != 0):
        n_chunks += 1
    return n_chunks

def extract_chunk_features(chunk):
    """
    Extracts the statistics features from the data in the chunk.

    Parameters
    ----------
    chunk : array
        An array of numbers.

    Returns
    -------
    out : array
        A tuple containing the mean, median, standard deviation, skewness,
        kurtosis, minimum, maximum and quartiles.
    """
    return (np.mean(chunk), np.median(chunk), np.std(chunk), stats.skew(chunk), stats.kurtosis(chunk),
            np.min(chunk), np.max(chunk), np.percentile(chunk, 25), np.percentile(chunk, 75))

# def extract_test_features(rates, c_vel, d_vel):
#     """
#     Extracts the statistics features from the arrays of the compression tests.
#
#     Parameters
#     ----------
#     rates : array
#         A buffer array with the compression rates tested.
#     c_vel : array
#         A buffer array with the compression velocities tested.
#     d_vel : array
#         A buffer array with the decompression velocities tested.
#
#     Returns
#     -------
#     out : array
#         An array containing the mean, standard deviation, quartiles, minimum
#         and maximum of each test array.
#     """
#     return [np.mean(rates), np.std(rates), np.percentile(rates, 25), np.percentile(rates, 75),
#             np.min(rates), np.max(rates), np.mean(c_vel), np.std(c_vel), np.percentile(c_vel, 25),
#             np.percentile(c_vel, 75), np.min(c_vel), np.max(c_vel), np.mean(d_vel), np.std(d_vel),
#             np.percentile(d_vel, 25), np.percentile(d_vel, 75), np.min(d_vel), np.max(d_vel)]

# FILENAMES = ['F:DADES/WRF_India.h5']
FILENAMES = ['WRF_India.h5', 'WRF_India-LSD3.h5', 'WRF_India-LSD2.h5', 'WRF_India-LSD1.h5',
             'GSSTF_NCEP.3-2000-zlib-5.h5']
BLOCK_SIZES = [2**13, 2**15, 2**17, 2**20, 0]
C_LEVELS = range(10)
COLS = ['Filename', 'DataSet', 'Chunk Number','Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max',
        'Q1', 'Q3', 'Block Size', 'Codec', 'Filter', 'CL', 'CRate', 'CSpeed', 'DSpeed']
blosc.set_nthreads(4)

if os.path.isfile('blosc_test_data.csv'):
    df = pd.read_csv('blosc_test_data.csv', sep='\t')
else:
    df = pd.DataFrame()

for filename in FILENAMES:
    for reading in file_reader(filename):
        n_chunks = calculate_nchunks(reading[1].dtype.itemsize, reading[1].size)
        print("Starting tests with", filename, reading[0])
        for i, chunk in enumerate(chunk_generator(reading[1])):
            chunk_features = extract_chunk_features(chunk)
            for block_size in BLOCK_SIZES:
                blosc.set_blocksize(block_size)
                for codec in blosc.compressor_list():
                    for filter in [blosc.NOSHUFFLE, blosc.SHUFFLE, blosc.BITSHUFFLE]:
                        for clevel in C_LEVELS:
                            row_data = (filename, reading[0], i + 1) + chunk_features +\
                                       (block_size / 2**10, codec, blosc.filters[filter], clevel) +\
                                        test_codec(chunk, codec, filter, clevel)
                            df = df.append(dict(zip(COLS, row_data)), ignore_index=True)
            print("%5.2f%% %-s %-s chunk %d completed" % ((i + 1)/n_chunks*100, filename, reading[0], (i + 1)))
            df.to_csv('blosc_test_data.csv', sep='\t', index=False)

# for block_size in BLOCK_SIZES:
#     blosc.set_blocksize(block_size)
#     data_features = extract_data_features(buffer)
#     count = 1
#     n_chunks = calculate_nchunks(buffer.dtype.itemsize, buffer.size)
#     # print("------------", chunk_size, filename.upper(), "------------")
#     for codec in blosc.compressor_list():
#         for filter in [blosc.NOSHUFFLE, blosc.SHUFFLE, blosc.BITSHUFFLE]:
#             for clevel in C_LEVELS:
#                 rates = np.empty(n_chunks, dtype=float)
#                 c_vel = np.empty(n_chunks, dtype=float)
#                 d_vel = np.empty(n_chunks, dtype=float)
#
#                 # TODO EXTRACT
#                 for i, chunk in enumerate(chunk_generator(buffer)):
#                     test = test_codec(chunk, codec, filter, clevel)
#                     rates[i], c_vel[i], d_vel[i] = test[0], test[1], test[2]
#                 print("%-10s  %5.2f%% %-s %-s %d" % ((codec + str(filter) + str(clevel)), count/180*100, filename, reading[0], chunk_size))
#                 count += 1
#                 test_features = extract_test_features(rates, c_vel, d_vel)
#                 row_data = np.append(np.append(heading, data_features), test_features)
#                 df = df.append(dict(zip(COLS, row_data)), ignore_index=True)
#                 # print('\n\nROW ADDED\n', df)
#                 df.to_csv('blosc_test_data.csv', sep='\t', index=False)
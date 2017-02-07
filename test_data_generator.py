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

def chunk_generator(buffer, size):
    """
    Given a buffer array generates data chunks of 2^(size) bytes.

    Parameters
    ----------
    buffer : array
        Buffer array of data

    Returns
    -------
    out : array
        A part of 2^(size) bytes of extracted from the original buffer.
    """
    mega = int((2 ** size) / buffer.dtype.itemsize)
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
    with tables.open_file(filename) as f:
        for child in f.root:
            if (child.size_in_memory > 10 * 2**20):
                yield (child._v_pathname, child[:].reshape(functools.reduce(lambda x, y: x * y, child.shape)))


def calculate_nchunks(chunk_size, type_size, buffer_size):
    """
    Calculates the number of chunks for the buffer.

    Parameters
    ----------
    chunk_size : int
        The power of the chunk size.
    type_size : int
        The type size in bytes.
    buffer_size : int
        The buffer size in number of elements.

    Returns
    -------
    out : int
        The number of chunks associated with the buffer and chunk size.
    """
    chunks_aux = int((2 ** chunk_size) / type_size)
    q, r = divmod(buffer_size, chunks_aux)
    n_chunks = q
    if (r != 0):
        n_chunks += 1
    return n_chunks

def extract_data_features(buffer):
    """
    Extracts the statistics features from the data in the buffer.

    Parameters
    ----------
    buffer : array
        A buffer array of numbers.

    Returns
    -------
    out : array
        An array containing the mean, median, standard deviation, skewness,
        kurtosis, minimum, maximum and quartiles.
    """
    return [np.mean(buffer), np.median(buffer), np.std(buffer), stats.skew(buffer), stats.kurtosis(buffer),
            np.min(buffer), np.max(buffer), np.percentile(buffer, 25), np.percentile(buffer, 75)]

def extract_test_features(rates, c_times, d_times):
    """
    Extracts the statistics features from the arrays of the compression tests.

    Parameters
    ----------
    rates : array
        A buffer array with the compression rates tested.
    c_times : array
        A buffer array with the compression times tested.
    d_times : array
        A buffer array with the decompression times tested.

    Returns
    -------
    out : array
        An array containing the mean, standard deviation, quartiles, minimum
        and maximum of each test array.
    """
    return [np.mean(rates), np.std(rates), np.percentile(rates, 25), np.percentile(rates, 75),
            np.min(rates), np.max(rates), np.mean(c_times), np.std(c_times), np.percentile(c_times, 25),
            np.percentile(c_times, 75), np.min(c_times), np.max(c_times), np.mean(d_times), np.std(d_times),
            np.percentile(d_times, 25), np.percentile(d_times, 75), np.min(d_times), np.max(d_times)]

# FILENAMES = ['F:DADES/WRF_India.h5']
FILENAMES = ['WRF_India.h5', 'WRF_India-LSD3.h5', 'WRF_India-LSD2.h5', 'WRF_India-LSD1.h5',
             'GSSTF_NCEP.3-2000-zlib-5.h5', '/home/francesc/AHMHighResPointCloud.f5']
CHUNK_SIZES = [15, 17, 19, 21]
C_LEVELS = range(10)
COLS = ['Filename', 'Dataset', 'Chunk size', 'Codec', 'Filter', 'CL','Mean', 'Median', 'Sd', 'Skew', 'Kurt',
        'Min', 'Max', 'Q1', 'Q3', 'CR_mean', 'CR_sd', 'CR_q1', 'CR_q3', 'CR_min', 'CR_max', 'CT_mean', 'CT_sd',
        'CT_q1', 'CT_q3', 'CT_min', 'CT_max', 'DT_mean', 'DT_sd', 'DT_q1', 'DT_q3', 'DT_min', 'DT_max']

if os.path.isfile('blosc_test_data.csv'):
    df = pd.read_csv('blosc_test_data.csv', sep='\t')
else:
    df = pd.DataFrame()

test_id = np.empty(6, dtype=object)
for filename in FILENAMES:
    test_id[0] = filename
    for reading in file_reader(filename):
        buffer = reading[1]
        test_id[1] = reading[0]
        for chunk_size in CHUNK_SIZES:
            test_id[2] = (2**chunk_size/2**10)
            data_features = extract_data_features(buffer)
            count = 1
            n_chunks = calculate_nchunks(chunk_size, buffer.dtype.itemsize, buffer.size)
            # print("------------", chunk_size, filename.upper(), "------------")
            for codec in blosc.compressor_list():
                test_id[3] = codec
                for filter in [blosc.NOSHUFFLE, blosc.SHUFFLE, blosc.BITSHUFFLE]:
                    test_id[4] = blosc.filters[filter]
                    for clevel in C_LEVELS:
                        test_id[5] = clevel
                        rates = np.empty(n_chunks, dtype=float)
                        c_times = np.empty(n_chunks, dtype=float)
                        d_times = np.empty(n_chunks, dtype=float)
                        for i, chunk in enumerate(chunk_generator(buffer, chunk_size)):
                            test = test_codec(chunk, codec, filter, clevel)
                            rates[i], c_times[i], d_times[i] = test[0], test[1], test[2]
                        print("%-10s  %5.2f%% %-s %-s %d" % ((codec + str(filter) + str(clevel)), count/180*100, filename, reading[0], chunk_size))
                        count += 1
                        test_features = extract_test_features(rates, c_times, d_times)
                        row_data = np.append(np.append(test_id, data_features), test_features)
                        df = df.append(dict(zip(COLS, row_data)), ignore_index=True)
                        # print('\n\nROW ADDED\n', df)
                        df.to_csv('blosc_test_data.csv', sep='\t', index=False)
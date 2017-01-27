from __future__ import print_function
import numpy as np
import time
import blosc
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D

def chunk_generator():
    for i in range( 63 ):
        yield buffer[ i*N: (i+1)*N ]

def test_codec( chunk, codec, filter, clevel ):
    """
        test_codec(chunk_generator, codec, filter, clevel)

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
            The associated compression time and rate.

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
    t0 = time.clock()
    c = blosc.compress_ptr( chunk.__array_interface__['data'][0],
                           chunk.size, chunk.dtype.itemsize,
                           clevel = clevel, shuffle = filter, cname = codec )
    tc = time.clock() - t0
    out = np.empty( chunk.size, dtype = chunk.dtype )
    t0 = time.clock()
    blosc.decompress_ptr( c, out.__array_interface__['data'][0] )
    td = time.clock() - t0
    rate = ( N * 8. / len(c) )
    assert ( ( chunk == out ).all() )
    # print( "  *** %-8s, %-10s *** %6.3f s / %5.3f s " %
    #        ( codec, blosc.filters[filter], tc, td), end='')
    # print( "\tCompr. ratio: %5.1fx" % rate )
    return ( tc, rate )

N = int(1e6)
N_CHUNKS = 63
buffer = np.empty(int(N_CHUNKS * 1e6), dtype=np.int64)
for i in range(1, N_CHUNKS+1 ):
    buffer[ (i-1)*N : (i)*N ] = np.random.randint( 0, 2**i, N, dtype = np.int64 )
C_LEVELS = (1,9)
CODECS = ('blosclz', 'lz4')

for codec in CODECS:
    for c_level in C_LEVELS:
        print('CORRELATIONS WITH ', codec.upper(), ' AND COMPRESSION LEVEL ', c_level)
        tblz = np.empty(N_CHUNKS, dtype=float)
        rblz = np.empty(N_CHUNKS, dtype=float)
        tzstd = np.empty(N_CHUNKS, dtype=float)
        rzstd = np.empty(N_CHUNKS, dtype=float)
        tlz4 = np.empty(N_CHUNKS, dtype=float)
        rlz4 = np.empty(N_CHUNKS, dtype=float)
        tlz4hc = np.empty(N_CHUNKS, dtype=float)
        rlz4hc = np.empty(N_CHUNKS, dtype=float)
        tsnappy = np.empty(N_CHUNKS, dtype=float)
        rsnappy = np.empty(N_CHUNKS, dtype=float)
        tzlib = np.empty(N_CHUNKS, dtype=float)
        rzlib = np.empty(N_CHUNKS, dtype=float)

        for i, chunk in enumerate(chunk_generator()):
            tblz[i], rblz[i] = test_codec(chunk, codec, blosc.SHUFFLE, c_level)
            tzstd[i], rzstd[i] = test_codec(chunk, 'zstd', blosc.SHUFFLE, 1)
            if (codec != 'lz4'):
                tlz4[i], rlz4[i] = test_codec(chunk, 'lz4', blosc.SHUFFLE, 1)
            else:
                tlz4[i], rlz4[i] = test_codec(chunk, 'blosclz', blosc.SHUFFLE, 1)
            tlz4hc[i], rlz4hc[i] = test_codec(chunk, 'lz4hc', blosc.SHUFFLE, 1)
            tsnappy[i], rsnappy[i] = test_codec(chunk, 'snappy', blosc.SHUFFLE, 1)
            tzlib[i], rzlib[i] = test_codec(chunk, 'zlib', blosc.SHUFFLE, 1)
        # PEARSON R
        print("-------COMPRESSION RATES---------")
        codec_aux = 'lz4'
        if (codec == 'lz4'):
            codec_aux = 'blosclz'
        print("Pearson ", codec, "-zstd: ", pearsonr(rblz, rzstd))
        print("Pearson ", codec, "-", codec_aux, ": ", pearsonr(rblz, rlz4))
        print("Pearson ", codec, "-lz4hc: ", pearsonr(rblz, rlz4hc))
        print("Pearson ", codec, "-snappy: ", pearsonr(rblz, rsnappy))
        print("Pearson ", codec, "-zlib: ", pearsonr(rblz, rzlib))
        print("-------COMPRESSION TIMES---------")
        print("Pearson ", codec, "-zstd: ", pearsonr(tblz, tzstd))
        print("Pearson ", codec, "-", codec_aux, ": ", pearsonr(tblz, tlz4))
        print("Pearson ", codec, "-lz4hc: ", pearsonr(tblz, tlz4hc))
        print("Pearson ", codec, "-snappy: ", pearsonr(tblz, tsnappy))
        print("Pearson ", codec, "-zlib: ", pearsonr(tblz, tzlib), '\n')



        # 2D GRAPHICS
        # f, axarr = plt.subplots(3,2)
        # axarr[0,0].scatter(rblz, rzstd, c=range(63), cmap='autumn_r')
        # axarr[0,0].set_title('RATES: blosclz9 VS zstd')
        #
        # axarr[0,1].scatter(tblz, tzstd, c=range(63), cmap='autumn_r')
        # axarr[0,1].set_title('TIMES: blosclz9 VS zstd')
        #
        # axarr[1,0].scatter(rblz, rlz4, c=range(63), cmap='autumn_r')
        # axarr[1,0].set_title('RATES: blosclz9 VS lz4')
        #
        # axarr[1,1].scatter(tblz, tlz4, c=range(63), cmap='autumn_r')
        # axarr[1,1].set_title('TIMES: blosclz9 VS lz4')
        #
        # axarr[2,0].scatter(rblz, rlz4hc, c=range(63), cmap='autumn_r')
        # axarr[2,0].set_title('RATES: blosclz9 VS lz4hc')
        #
        # axarr[2,1].scatter(tblz, tlz4hc, c=range(63), cmap='autumn_r')
        # axarr[2,1].set_title('TIMES: blosclz9 VS lz4hc')
        #
        # f2, axarr2 = plt.subplots(2,2)
        # axarr2[0,0].scatter(rblz, rsnappy, c=range(63), cmap='autumn_r')
        # axarr2[0,0].set_title('RATES: blosclz9 VS snappy')
        #
        # axarr2[0,1].scatter(tblz, tsnappy, c=range(63), cmap='autumn_r')
        # axarr2[0,1].set_title('TIMES: blosclz9 VS snappy')
        #
        # axarr2[1,0].scatter(rblz, rzlib, c=range(63), cmap='autumn_r')
        # axarr2[1,0].set_title('RATES: blosclz9 VS zlib')
        #
        # axarr2[1,1].scatter(tblz, tzlib, c=range(63), cmap='autumn_r')
        # axarr2[1,1].set_title('TIMES: blosclz9 VS zlib')

        # 3D GRAPHICS
        # fig = plt.figure()
        # ax = fig.add_subplot(121, projection='3d')
        # ax.scatter(rblz, tblz, rzstd, c=range(63), label='C-Level (1,1)', cmap='autumn_r')
        # ax.legend()
        # ax.set_title('blosclz-9 VS zstd')
        #
        # ax2 = fig.add_subplot(122, projection='3d')
        # ax2.scatter(rblz, tblz, tzstd, c=range(63), label='C_Level (1,1)', cmap='autumn_r')
        # ax2.legend()
        # ax2.set_title('blosclz-9 VS zstd')

        # plt.show()
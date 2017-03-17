# IMPORTS, GLOBAL VARIABLES AND FUNCTION DEFINITION
import itertools
from turtledemo.__main__ import font_sizes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats.stats import pearsonr

# DICTIONARIES FOR BUILDING PERSONALIZED GRAPHS
COLOR_PALETTE = {'blosclz': '#5AC8FA', 'lz4': '#4CD964', 'lz4hc': '#FF3B30', 'snappy': '#FFCC00',
                 'zlib': '#FF9500', 'zstd': '#5856D6'}
MARKER_DICT = {'noshuffle': 'o', 'shuffle': 'v', 'bitshuffle': 's'}

# DIFFERENT COLUMN LISTS FROM THE DATAFRAME FOR SELECTING SPECIFIC INFO
# TODO N_STREAKS
COLS = ['Filename', 'DataSet', 'Table', 'DType', 'Chunk_Number', 'Chunk_Size', 'Mean', 'Median', 'Sd', 'Skew', 'Kurt',
        'Min', 'Max', 'Q1', 'Q3', 'Block_Size', 'Codec', 'Filter', 'CL', 'CRate', 'CSpeed', 'DSpeed']  # 'N_Streaks']
DESC_SET = ['DataSet', 'DType', 'Table', 'Chunk_Size']
CHUNK_FEATURES = ['Chunk_Size', 'Mean', 'Median', 'Sd', 'Skew', 'Kurt', 'Min', 'Max', 'Q1', 'Q3']  # 'N_Streaks']
TEST_FEATURES = ['CRate', 'CSpeed', 'DSpeed']
ALL_FEATURES = CHUNK_FEATURES + TEST_FEATURES

# AUX LIST FOR SELECTING DATAFRAMES
TYPES = ['float', 'int', 'str']
BLOCK_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# PATH TO FIGURES
FIG_PATH = '../figures/Blosc-test-analysis--'


# AUX FUNCTIONS
def outlier_lim(data):
    """Return the outliers limits."""

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    dif = q3 - q1

    return q1 - 1.5 * dif, q3 + 1.5 * dif, dif


# PLOTTING FUNCTIONS
def custom_boxplot(ax, y, title='', ylabel=''):
    """Customized boxplot style."""

    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='#FF3B30', markersize=8)
    medianprops = {'color': '#4CD964', 'linewidth': 3}
    boxprops = {'edgecolor': 'black', 'linestyle': '-', 'facecolor': '#EFEFF4'}
    whiskerprops = {'color': 'black', 'linestyle': '-'}
    capprops = {'color': 'black', 'linestyle': '-'}
    flierprops = {'color': 'black', 'marker': 'o'}

    ax.boxplot(y,
               medianprops=medianprops,
               boxprops=boxprops,
               whiskerprops=whiskerprops,
               capprops=capprops,
               flierprops=flierprops,
               meanprops=meanpointprops,
               meanline=False,
               showmeans=True,
               widths=0.5,
               patch_artist=True)

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    return ax


def custom_centered_scatter(ax, x, y):
    """Customized scatter plot ignoring outliers."""

    x_lim = outlier_lim(x)
    y_lim = outlier_lim(y)
    x_, y_ = [], []
    for i in range(len(x)):
        if x_lim[0] <= x.iloc[i] <= x_lim[1] and y_lim[0] <= y.iloc[i] <= y_lim[1]:
            x_.append(x.iloc[i])
            y_.append(y.iloc[i])
    ax.scatter(x_, y_, color='#007AFF', marker='.', linewidth=3)
    if x_lim[2] > 0 and y_lim[2] > 0:
        ax.set_xlim(x_lim[0:2])
        ax.set_ylim(y_lim[0:2])
        fit = np.polyfit(x_, y_, deg=1)
        x_ = np.sort(x_)
        ax.plot(x_, fit[0] * x_ + fit[1], color='#FF3B30', linewidth=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def custom_pearson_scatter(ax, x_, y_, c_, m_, title, dtype):
    """Customized scatter plot for correlations."""

    for x, y, c, m, size in zip(x_, y_, c_, m_, np.asarray(list((itertools.repeat([10, 16], len(c_))))).flatten()):
        ax.plot(x, y, alpha=0.8, c=c, marker=m, linewidth=3, markersize=size)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='#CECED2')
    ax.xaxis.grid(color='#CECED2')
    ax.set_xlabel('Pearson C.Rates')
    ax.set_ylabel('Pearson C.Speeds')
    ax.set_title(title + ' - PEARSON - ' + dtype.upper())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def custom_sc_legend(fig):
    """Customized legend for the correlation plots."""

    handles = [mlines.Line2D([], [], color=color, marker='o', linestyle=' ',
                             markersize=10, label=label) for label, color in COLOR_PALETTE.items()]
    handles += [mlines.Line2D([], [], color='k', marker='o', linestyle=' ',
                              markersize=10, label='NOSHUFFLE'),
                mlines.Line2D([], [], color='k', marker='v', linestyle=' ',
                              markersize=10, label='SHUFFLE'),
                mlines.Line2D([], [], color='k', marker='s', linestyle=' ',
                              markersize=10, label='BITSHUFFLE'),
                mlines.Line2D([], [], color='k', marker='o', linestyle=' ',
                              markersize=10, label='Compression Level 1'),
                mlines.Line2D([], [], color='k', marker='o', linestyle=' ',
                              markersize=16, label='Compression Level 9')]
    labels = [label for label in COLOR_PALETTE.keys()]
    labels += ['NOSHUFFLE', 'SHUFFLE', 'BITSHUFFLE', 'Compression Level 1', 'Compression Level 9']
    fig.legend(handles=handles, labels=labels, loc='lower left', ncol=2, bbox_to_anchor=(1, 0.05))
    fig.tight_layout()

    return fig


def custom_lineplot_tests(ax, x, rates, c_speeds, d_speeds, title='', cl_mode=False):
    """Customized line plot for blosc test data."""

    ax.plot(x, rates, color='#007AFF', marker='o', markersize=8, linewidth=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='#CECED2')
    ax.xaxis.grid(color='#CECED2')
    ax.set_ylabel('Compression Rate', color='#007AFF')
    ax.tick_params('y', colors='#007AFF')
    blue_line = mlines.Line2D([], [], color='#007AFF', marker='o',
                              markersize=8, label='Compression ratio')
    plt.legend(handles=[blue_line], loc=2, bbox_to_anchor=(0., 1.01, 0., .102))
    ax2 = ax.twinx()
    ax2.plot(x, c_speeds, color='#FF3B30', marker='o', markersize=8, linewidth=3)
    ax2.plot(x, d_speeds, color='#4CD964', marker='o', markersize=8, linewidth=3)
    ax2.set_ylabel('Speed (GB/s)', color='k')
    ax.set_yticks(np.linspace(ax.get_ybound()[0], ax.get_ybound()[1], 6))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 6))
    red_line = mlines.Line2D([], [], color='#FF3B30', marker='o',
                             markersize=8, label='Compression')
    green_line = mlines.Line2D([], [], color='#4CD964', marker='o',
                               markersize=8, label='Decompression')
    plt.legend(handles=[red_line, green_line], loc=1, bbox_to_anchor=(0., 1.01, 1., .102))
    if not cl_mode:
        ax.set_xscale('log', basex=2)
        ax.set_xticks(x)
        ax.set_xticklabels(['Auto', '8K', '16K', '32K', '64K', '128K', '256K', '512K', '1MB', '2MB'])
        ax.set_xlabel('Block Size')
    else:
        ax.set_xticks(x)
        ax.set_xlabel('Compression level')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    return ax


def boxplot_data_builder(df):
    """Build data for the boxplots."""

    rates = []
    c_speeds = []
    d_speeds = []
    indices = []
    for i in range(3):
        if i == 2:
            dfaux = df[df.DType.str.contains('U') | df.DType.str.contains('S')]
        else:
            dfaux = df[df.DType.str.contains(TYPES[i])]
        if dfaux.size > 0:
            rates.append([dfaux['CRate']])
            c_speeds.append([dfaux['CSpeed']])
            d_speeds.append([dfaux['DSpeed']])
            indices.append(i)
    return rates, c_speeds, d_speeds, indices


def paint_dtype_boxplots(df):
    """Paint boxplots structured by dtype."""

    rates, c_speeds, d_speeds, indices = boxplot_data_builder(df)
    n = len(rates)
    if n > 1:
        fig = plt.figure(figsize=(20, 24))
        for i in range(n * 3):
            aux = 300 + n * 10 + i + 1
            pos = i % n
            ax = fig.add_subplot(aux)
            if i < n:
                custom_boxplot(ax, rates[pos], 'C.Rates-' + TYPES[indices[pos]],
                               'Compression Rates')
            elif i < n * 2:
                custom_boxplot(ax, c_speeds[pos], 'C.Speeds-' + TYPES[indices[pos]],
                               'Compression Speeds (GB/s)')
            else:
                custom_boxplot(ax, d_speeds[pos], 'D.Speeds-' + TYPES[indices[pos]],
                               'Decompression Speeds (GB/s)')
    else:
        fig = plt.figure(figsize=(20, 8))
        custom_boxplot(fig.add_subplot(131), rates[0],
                       'C.Rates-' + TYPES[indices[0]], 'Compression Rates')
        custom_boxplot(fig.add_subplot(132), c_speeds[0],
                       'C.Speeds-' + TYPES[indices[0]], 'Compression Speeds (GB/s)')
        custom_boxplot(fig.add_subplot(133), d_speeds[0],
                       'D.Speeds-' + TYPES[indices[0]], 'Decompression Speeds (GB/s)')
    fig.suptitle('Test features boxplots')
    plt.savefig(FIG_PATH + 'Test features boxplots' + '.png', bbox_inches='tight')


def block_cor_data_builder(df, onlystr, cl_mode):
    """Build data for the block correlation graphs"""

    rates = []
    c_speeds = []
    d_speeds = []
    indices = []
    block_values = [0] + BLOCK_SIZES
    if onlystr:
        options = [2]
    else:
        options = range(3)
    for i in options:
        if i == 2:
            dfaux = df[df.DType.str.contains('U') | df.DType.str.contains('S')]
        else:
            dfaux = df[df.DType.str.contains(TYPES[i])]
        if dfaux.size > 0:
            if not cl_mode:
                rates.append([dfaux[dfaux.Block_Size == size]['CRate'].mean() for size in block_values])
                c_speeds.append([dfaux[dfaux.Block_Size == size]['CSpeed'].mean() for size in block_values])
                d_speeds.append([dfaux[dfaux.Block_Size == size]['DSpeed'].mean() for size in block_values])
                indices.append(i)
            else:
                rates.append([dfaux[dfaux.CL == cl]['CRate'].mean() for cl in list(range(10))[1:]])
                c_speeds.append([dfaux[dfaux.CL == cl]['CSpeed'].mean() for cl in list(range(10))[1:]])
                d_speeds.append([dfaux[dfaux.CL == cl]['DSpeed'].mean() for cl in list(range(10))[1:]])
                indices.append(i)

    return rates, c_speeds, d_speeds, indices


def paint_block_cor(df, title='', onlystr=False, cl_mode=False):
    """Paint custom lineplots structured by dtype."""

    rates, c_speeds, d_speeds, indices = block_cor_data_builder(df, onlystr, cl_mode)
    if not cl_mode:
        x = [1] + BLOCK_SIZES
    else:
        x = list(range(10))[1:]
    n = len(rates)
    fig = plt.figure(figsize=(20, 8))
    sup_title = 'Block Size'
    if cl_mode:
        sup_title = 'Compression Level'
    fig.suptitle(sup_title + ' comparison with ' + title, fontsize=16)
    for i in range(n):
        pos = 100 + n * 10 + i + 1
        ax = fig.add_subplot(pos)
        custom_lineplot_tests(ax, x, rates=rates[i], c_speeds=c_speeds[i],
                              d_speeds=d_speeds[i], title='dtype - ' + TYPES[indices[i]], cl_mode=cl_mode)
    if n > 1:
        fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(FIG_PATH + sup_title + ' comparison with ' + title + '.png', bbox_inches='tight')


def paint_all_block_cor(df, filter_name, c_level=5, cl_mode=False, block_size=0):
    """Paint all custom lineplots for filter."""

    if filter_name == 'noshuffle':
        onlystr = True
    else:
        onlystr = False
    for codec in df.drop_duplicates(subset=['Codec'])['Codec']:
        if codec == 'blosclz' and filter_name == 'shuffle':
            if not cl_mode:
                paint_block_cor(df[(df.CL == c_level) & (df.Codec == codec) & (df.Filter == 'bitshuffle')],
                                codec.upper() + '-BITSHUFFLE-CL' + str(c_level), onlystr, cl_mode)
            else:
                paint_block_cor(
                    df[(df.Block_Size == block_size) & (df.Codec == codec) & (df.Filter == 'bitshuffle')],
                    codec.upper() + '-BITSHUFFLE-BLOCK' + str(block_size), onlystr, cl_mode)
        if not cl_mode:
            paint_block_cor(df[(df.CL == c_level) & (df.Codec == codec) & (df.Filter == filter_name)],
                            codec.upper() + '-' + filter_name.upper() + '-CL' + str(c_level), onlystr, cl_mode)
        else:
            paint_block_cor(
                df[(df.Block_Size == block_size) & (df.Codec == codec) & (df.Filter == filter_name)],
                codec.upper() + '-' + filter_name.upper() + '-BLOCK' + str(block_size), onlystr, cl_mode)


def paint_cl_comparison(df, filter_name, codec):
    """Paint custom plots comparing compression levels and block sizes."""

    data = []
    c_levels = [1, 3, 6, 9]
    for c_level in c_levels:
        data.append(block_cor_data_builder(df[(df.CL == c_level) & (df.Codec == codec) & (df.Filter == filter_name)],
                                           False, False))
    block_sizes = [1] + BLOCK_SIZES
    n = len(data[0][0])
    for i in range(n):
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Compression Level and block size comparison ' + codec.upper() + '-' +
                     TYPES[data[0][3][i]].upper(), fontsize=16)
        for j in range(4):
            pos = 200 + 20 + j + 1
            ax = fig.add_subplot(pos)
            custom_lineplot_tests(ax, block_sizes, data[j][0][i], data[j][1][i], data[j][2][i],
                                  title='C-Level ' + str(c_levels[j]))
        if n > 1:
            fig.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.2)
        plt.savefig(FIG_PATH + 'Compression Level and block size comparison ' + codec.upper() + '-' +
                    TYPES[data[0][3][i]].upper() + '.png', bbox_inches='tight')


def pearson_cor_data_builder(df, cname, clevel):
    """Build data for codec correlation graphs."""

    pearson_rates = [[], [], []]
    pearson_c_speeds = [[], [], []]
    codecs_cl = [[], [], []]
    colors = [[], [], []]
    markers = [[], [], []]
    for codec in df.drop_duplicates(subset=['Codec'])['Codec']:
        for filt in ['noshuffle', 'shuffle', 'bitshuffle']:
            df_blz1 = df[(df.Codec == cname) & (df.CL == clevel) & (df.Filter == 'noshuffle')]
            for c_level in [1, 9]:
                df_codec = df[(df.Codec == codec) & (df.CL == c_level) & (df.Filter == filt)]
                for i in range(3):
                    if i == 2:
                        dfaux = df_codec[df_codec.DType.str.contains('U') | df_codec.DType.str.contains('S')]
                        df_blz_aux = df_blz1[df_blz1.DType.str.contains('U') | df_blz1.DType.str.contains('S')]
                    else:
                        dfaux = df_codec[df_codec.DType.str.contains(TYPES[i])]
                        df_blz_aux = df_blz1[df_blz1.DType.str.contains(TYPES[i])]
                    if dfaux.size > 0:
                        pearson_rates[i].append(pearsonr(df_blz_aux['CRate'], dfaux['CRate']))
                        pearson_c_speeds[i].append(pearsonr(df_blz_aux['CSpeed'], dfaux['CSpeed']))
                        codecs_cl[i].append(codec + '-' + filt + '-' + str(c_level) + '-' + TYPES[i])
                        colors[i].append(COLOR_PALETTE[codec])
                        markers[i].append(MARKER_DICT[filt])

    return pearson_rates, pearson_c_speeds, codecs_cl, colors, markers


def paint_codec_pearson_corr(df, cname, clevel):
    """Paint custom graphs for codec correlation."""

    pearson_rates, pearson_c_speeds, codecs_cl, colors, markers = pearson_cor_data_builder(df, cname, clevel)

    for i in range(3):
        if len(pearson_rates[i]) > 0:
            fig = plt.figure(figsize=(10, 9))
            ax = fig.add_subplot(111)
            custom_pearson_scatter(ax, [x[0] for x in pearson_rates[i]],
                                   [x[0] for x in pearson_c_speeds[i]],
                                   colors[i], markers[i], cname.upper() + '-CL' + str(clevel), TYPES[i])
            custom_sc_legend(fig)
            plt.savefig(FIG_PATH + ax.get_title() + '.png', bbox_inches='tight')


def custom_pairs(df, col_names):
    """Paint scatter matrix plot."""

    print('%d points' % df.shape[0])
    fig, axs = plt.subplots(3, len(col_names), sharex='col', sharey='row')
    fig.set_size_inches(20, 12)
    for j, y in enumerate(['CRate', 'CSpeed', 'DSpeed']):
        for i, x in enumerate(col_names):
            custom_centered_scatter(axs[j, i], df[x], df[y])
            if j == 2:
                axs[j, i].set_xlabel(x)
            if i == 0:
                axs[j, i].set_ylabel(y)
    fig.tight_layout()
    fig.suptitle(str(col_names) + ' VS Test Features', fontsize=16)
    plt.subplots_adjust(top=0.95, hspace=0.01, wspace=0)
    plt.savefig(FIG_PATH + str(col_names) + 'VS Test Features' + '.png', bbox_inches='tight')

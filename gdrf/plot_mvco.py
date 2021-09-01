import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Optional
from datetime import datetime
import os
GROUND_TRUTH_FILE = '../data/count_by_class_time_seriesCNN_hourly19Aug2021.csv'

def read_the_csv(file, start_col=1, bad_cols = None):
    if bad_cols is None:
        bad_cols = {}
    df = pd.read_csv(file)
    df[df.columns[0]] = df[df.columns[0]].astype('datetime64')
    df.index = df[df.columns[0]]
    df = df[df.columns[start_col:]]
    good_columns = [c for c in df.columns if c not in bad_cols]
    df = df[good_columns]
    df += 1e-10
    df = df.div(df.sum(axis=1), axis=0)
    return df

def make_stackplot(df: pd.DataFrame, fig: plt.Figure, ax: plt.Axes, title: str = "", labels: Optional[list[str]] = None):
    x = df.index
    ys = df.to_numpy().T
    if labels is not None:
        ax.stackplot(x, ys, labels=labels)
    else:
        ax.stackplot(x, ys)
    ax.set_title(title)

def make_cellplot(df: pd.DataFrame, fig: plt.Figure, ax: plt.Axes, title: str = "", xticklabels=False):
    data_numpy = df.to_numpy()
    xs = df.index
    nt, nc = data_numpy.shape
    time_coords, cat_coords = np.mgrid[0:nt:1, 0:nc:1]
    pcm = ax.pcolor(time_coords, cat_coords, data_numpy,
                    norm=colors.LogNorm(vmin=0.00001, vmax=1.0), shading='auto',
                    )#linewidths=2, edgecolors='k')
    min_t = min(xs)
    max_t = max(xs)
    years = [datetime(year=x, month=1, day=1) for x in range(2000, 2050) if min_t <= datetime(year=x, month=1, day=1) <= max_t]
    year_coords = [(y - min_t) / (max_t - min_t) for y in years]
    year_coords = [x*nt for x in year_coords]
    ax.set_xticks(year_coords)
    ax.set_xticklabels(["" for _ in years])
    if xticklabels:
        ax.set_xticklabels([y.strftime('%Y') for y in years])
    ax.set_title(title)
    ax.set_yticks(list(range(nc)))
    ax.set_yticklabels(df.columns)
    plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
    fontsizeticklabels = ax.get_xticklabels()
    if len(ax.get_yticklabels()) < 20:
        fontsizeticklabels += ax.get_yticklabels()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + fontsizeticklabels):
        item.set_fontsize(100)

    return pcm

def make_wt_plot(word_topic_matrix: pd.DataFrame, fig: plt.Figure, ax: plt.Axes, columns, title: str = "Word-Topic Matrix"):
    K, V = word_topic_matrix.shape
    topic_coords, word_coords = np.mgrid[0:K:1, 0:V:1]
    pcm = ax.pcolor(word_coords.T, topic_coords.T, word_topic_matrix.T,
                    norm=colors.LogNorm(vmin=0.00001, vmax=1.0), shading='auto',
                    linewidths=2, edgecolors='k')
    ax.set_title('Inferred Word-Topic Matrix')
    ax.set_yticks(list(range(K)))
    ax.set_yticklabels([f'Topic {i}' for i in range(1, K + 1)])
    ax.set_xticks(list(range(V)))
    ax.set_xticklabels(columns)
    plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
    fig.colorbar(pcm, ax=ax, extend='min')

def make_plots(folder: str):
    full_folder = os.path.join('..', 'data', folder)
    topic_prob_fn = os.path.join(full_folder, '_'.join([folder, 'topic_probs.csv']))
    word_prob_fn = os.path.join(full_folder, '_'.join([folder, 'word_probs.csv']))
    word_topic_matrix_fn = os.path.join(full_folder, '_'.join([folder, 'word_topic_matrix.csv']))
    bad_cols = {
        'amoeba',
        'bead',
        'bubble',
        'camera_spot',
        'ciliate',
        'coccolithophorid',
        'cryptophyta',
        'detritus',
        'detritus_clear',
        'fecal_pellet',
        'fiber',
        'fiber_TAG_external_detritus',
        'flagellate',
        'flagellate_morphotype1',
        'flagellate_morphotype3',
        'mix',
        'pennate',
        'pennate_Pseudo-nitzschia',
        'pennate_Thalassionema',
        'pennate_morphotype1',
        'pollen',
        'shellfish_larvae',
        'square_unknown',
        'unknown2',
        'zooplankton',
    }
    ground_truth = read_the_csv(GROUND_TRUTH_FILE, start_col=2, bad_cols=bad_cols)

    topic_prob = read_the_csv(topic_prob_fn)
    word_prob = read_the_csv(word_prob_fn)
    word_topic_matrix = pd.read_csv(word_topic_matrix_fn, index_col=0).to_numpy()

    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(60.5, 60.5)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    make_cellplot(ground_truth, fig, axes[0], title="Grouth Truth Taxa")
    make_cellplot(topic_prob, fig, axes[1], title="MAP Topic Probabilities")
    pcm = make_cellplot(word_prob, fig, axes[2], xticklabels=True, title="MAP Word Probabilities")
    fig.colorbar(pcm, cax=cbar_ax, extend='min')
    for tick in cbar_ax.get_yticklabels():
        tick.set_fontsize(100)
    plt.show()
    fig.savefig(os.path.join(full_folder, '_'.join([folder, 'time_series_cell.png'])), dpi=100)

    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(10.5, 10.5)
    make_stackplot(ground_truth, fig, axes[0], title="Ground Truth Taxa")
    make_stackplot(topic_prob, fig, axes[1], title="MAP Topic Probabilities", labels=topic_prob.columns)
    make_stackplot(word_prob, fig, axes[2], title="MAP Word probabilities")
    plt.show()
    fig.savefig(os.path.join(full_folder, '_'.join([folder, 'time_series.png'])), dpi=100)

    fig, ax = plt.subplots()
    fig.set_size_inches(60.5, 5.5)
    make_wt_plot(word_topic_matrix=word_topic_matrix, fig=fig, ax=ax, columns=ground_truth.columns, title="Word-Topic Matrix")
    plt.show()
    fig.savefig(os.path.join(full_folder, '_'.join([folder, 'wt_matrix.png'])), dpi=500, bbox_inches='tight')


if __name__ == "__main__":
    make_plots("aug_24_21_mvco_6_0.1_0.2_600_0.01_25000_0.001_3_57575")


import pandas as pd
import numpy as np
import torch
import csv

import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize, datashade
import datashader as ds
from holoviews.operation import decimate
import hvplot.pandas
hv.extension('matplotlib')
import panel as pn
pn.extension()

from typing import Union, Optional

def parse_matrix(
    data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray]
):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().to_numpy()
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if isinstance(data, str):
        data = pd.read_csv(data, header=0, index_col=0)
    return data

def parse_spatiotemporal(
    data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray],
    index: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, pd.Index, str]] = None,
    dims: Optional[int] = None
):
    dims = dims if dims is not None else 1

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().to_numpy()
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if isinstance(index, str):
        index = pd.read_csv(index, header=0)
    if isinstance(index, torch.Tensor):
        index = index.detach().cpu().to_numpy()
    if not ((index is None) or isinstance(index, pd.Index)):
        index = pd.Index(index)
    if isinstance(data, str):
        if index is None:
            data = pd.read_csv(data, header=0, index_col=list(range(dims)), parse_dates=True)
        else:
            data = pd.read_csv(data, header=0)
            data = data.set_index(index)
    return data


def stackplot_1d(
    data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray],
    index: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, pd.Index, str]] = None,
):
    data = parse_spatiotemporal(data, index, dims=1)
    data = data.div(data.sum(axis=1), axis=0)
    data = data[~data.index.duplicated()]
    # if len(data) > 1000:  # Too many points
    #     n = len(data) // 1000
    #     data = data[np.arange(len(data)) % n == 1]
    areas = []
    vdim = hv.Dimension('probability', label='Probability')
    kdim = hv.Dimension(data.index.name, label=data.index.name)
    for c in data.columns:
        d = data[c]
        d.name = 'probability'
        a = hv.Area(d, label=str(c), kdims=kdim, vdims=vdim).opts(
            linewidth=0, color=hv.Cycle('tab20'), aspect=2, fig_inches=15, fig_bounds=(0, 0, 1, 1), fig_size=400)
        areas.append(a)
    overlay = hv.Overlay(areas)
    stack = hv.Area.stack(overlay)
    return stack

def matrix_plot(
    data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray],
    log: bool = False,
    epsilon: float = 1e-10
):
    data = parse_matrix(data)
    if log:
        data = data + epsilon
    heatmap = data.hvplot.heatmap()
    heatmap.opts(opts.HeatMap(colorbar=True, clim=(1e-5, 1.), logz=log, cmap='summer', fig_size=400
    ))
    return heatmap


def display(plot):
    pn.Row(plot).show()


def stackplot_1d_cli(
    data: str,
    index: str = None
):
    """
    Creates a 1-D stackplot from the probabilities in the CSV file `data`.
    Optionally, a single-column CSV file `index` may be provided.
    The first row of CSV files should be a header.
    All columns are considered data, except if no `index` CSV is provided the first column of `data` is the index.

    :param str data: A CSV file with probabilities. Header row is required, index column is optional
    :param str index: An optional CSV file with the index. Header row is required.
    """
    display(stackplot_1d(data, index))


def matrixplot_cli(
    data: str,
    log: bool = False,
    epsilon: float = 1e-10
):
    """
    Creates a matrix plot from the probabilities in the CSV file `data`.
    The first row of the CSV file should be a header.
    The first column of the CSV file should be an index.
    All other columns are considered data.
    Optionally, the probabilities may be plotted on a log-scale colorbar.

    :param str data: A CSV file with probabilities. Header row is required, index column is required.
    :param bool log: An optional CSV file with the index. Header row is required.
    :param float epsilon: An optional "jitter" parameter, for plotting small probabilities on a log-scale
    """
    display(matrix_plot(data, log=log, epsilon=epsilon))

def main():
    # file = "/home/sansoucie/PycharmProjects/gdrf/data/test_matrix.csv"
    # plot = matrix_plot(data=file, log=True)
    file = "/home/sansoucie/PycharmProjects/gdrf/data/data.csv"
    plot = stackplot_1d(data=file)
    row = pn.Row(plot)
    row.show()


if __name__ == "__main__":
    main()





from typing import Optional, Tuple, Union

import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
import torch
from holoviews import opts

hv.extension("matplotlib")
pn.extension()
pd.options.plotting.backend = "holoviews"


def parse_matrix(data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray]):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().to_numpy()
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if isinstance(data, str):
        data = pd.read_csv(data, header=0, index_col=0)
    return data


def parse_spatiotemporal(
    data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray],
    index: Optional[
        Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, pd.Index, str]
    ] = None,
    dims: Optional[int] = None,
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
            data = pd.read_csv(
                data, header=0, index_col=list(range(dims)), parse_dates=True
            )
        else:
            data = pd.read_csv(data, header=0)
            data = data.set_index(index)
    if data.index.name is None:
        data.index.names = ["index"]
    return data


def _normalize(data: pd.DataFrame):
    ret = data.div(data.sum(axis=1), axis=0)
    return ret[~ret.index.duplicated()]


def stackplot_1d(
    data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray],
    index: Optional[
        Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, pd.Index, str]
    ] = None,
    legend=False,
):
    data = _normalize(parse_spatiotemporal(data, index, dims=1))
    # if len(data) > 1000:  # Too many points
    #     n = len(data) // 1000
    #     data = data[np.arange(len(data)) % n == 1]
    areas = []
    vdim = hv.Dimension("probability", label="Probability")
    kdim = hv.Dimension(data.index.name, label=data.index.name)
    for c in data.columns:
        d = data[c]
        d.name = "probability"
        a = hv.Area(d, label=str(c), kdims=kdim, vdims=vdim).opts(
            linewidth=0,
            color=hv.Cycle("tab20"),
            aspect=2,
            fig_inches=15,
            fig_bounds=(0, 0, 1, 1),
            fig_size=400,
            show_legend=legend,
        )
        areas.append(a)
    overlay = hv.Overlay(areas)
    stack = hv.Area.stack(overlay).opts(legend_position="right")
    return stack


def _heatmap(
    data: pd.DataFrame,
    fig_inches: Tuple[float, float],
    log: bool = False,
    probability: bool = False,
):
    heatmap = data.hvplot.heatmap()

    hmopts = (lambda **x: x)(
        logz=log,
        show_values=False,
        xrotation=90,
        fig_size=400,
        cmap="summer" if probability else "prism",
    )
    if probability:
        hmopts["clim"] = (1e-5, 1.0)
        hmopts["colorbar"] = True
        hmopts["fig_inches"] = fig_inches
        hmopts["aspect"] = 6

    heatmap.opts(opts.HeatMap(**hmopts))
    return heatmap


def maxplot_2d(
    data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray],
    index: Optional[
        Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, pd.Index, str]
    ] = None,
):
    data = _normalize(parse_spatiotemporal(data, index, dims=2))
    mle = data.idxmax(axis=1).unstack(level=1).astype(int)
    fig_inches = (0.01 * len(data.columns), 0.01 * len(data))
    return _heatmap(mle, fig_inches=fig_inches, log=False, probability=False)


def matrix_plot(
    data: Union[str, torch.Tensor, pd.DataFrame, np.ndarray],
    log: bool = False,
    epsilon: float = 1e-10,
):
    data = parse_matrix(data)
    if log:
        data = data + epsilon
    fig_inches = (0.125 * len(data.columns), 0.25 * len(data))
    return _heatmap(data, fig_inches=fig_inches, log=log, probability=True)


def display(plot):
    pn.Row(plot).show()


def stackplot_1d_cli(data: str, index: str = None, legend: bool = False):
    """
    Creates a 1-D stackplot from the probabilities in the CSV file `data`.
    Optionally, a single-column CSV file `index` may be provided.
    The first row of CSV files should be a header.
    All columns are considered data, except if no `index` CSV is provided the first column of `data` is the index.

    :param str data: A CSV file with probabilities. Header row is required, index column is optional
    :param str index: An optional CSV file with the index. Header row is required.
    :param bool legend: Whether or not to display a legend
    """
    display(stackplot_1d(data, index, legend=legend))


def matrixplot_cli(data: str, log: bool = False, epsilon: float = 1e-10):
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


def maxplot_2d_cli(data: str, index: str = None):
    """
    Creates a 2-D maximum plot from the probabilities in the CSV file `data`.
    Optionally, a two-column CSV file `index` may be provided.
    The first row of CSV files should be a header.
    All columns are considered data, except if no `index` CSV is provided the first two columns of `data` are the index.

    :param str data: A CSV file with probabilities. Header row is required, index columns are optional
    :param str index: An optional CSV file with the index. Header row is required.
    """
    display(maxplot_2d(data, index))


# def main():
#     # file = "/home/sansoucie/PycharmProjects/gdrf/data/data_2d_artificial.csv"
#     # plot = maxplot_2d(data=file)
#     file = "/home/sansoucie/PycharmProjects/gdrf/data/data.csv"
#     plot = stackplot_1d(data=file, legend=True)
#     row = pn.Row(plot)
#     row.show()
#
#
# if __name__ == "__main__":
#     main()

"""Console script for gdrf."""

import fire
from train_script import train
from visualize import matrixplot_cli, maxplot_2d_cli, stackplot_1d_cli

import gdrf


def help():
    print("gdrf")
    print("=" * len("gdrf"))
    print("Pytorch+GPytorch implementation of GDRFs from San Soucie et al. 2020")


class PlotCLI(object):
    def __init__(self):
        self.stack = stackplot_1d_cli
        self.matrix = matrixplot_cli
        self.max = maxplot_2d_cli


class CLI(object):
    def __init__(self):
        self.help = help
        self.train = train
        self.plot = PlotCLI()


def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()  # pragma: no cover

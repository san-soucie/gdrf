"""Console script for gdrf."""

import fire
from .train import train

def help_fire():
    print("gdrf")
    print("=" * len("gdrf"))
    print("Pytorch+GPytorch implementation of GDRFs from San Soucie et al. 2020")


def main():
    fire.Fire({"help": help_fire,
               "train": train})


if __name__ == "__main__":
    main()  # pragma: no cover

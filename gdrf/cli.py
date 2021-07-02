"""Console script for gdrf."""

import fire


def help():
    print("gdrf")
    print("=" * len("gdrf"))
    print("Pytorch+GPytorch implementation of GDRFs from San Soucie et al. 2020")


def main():
    fire.Fire({"help": help})


if __name__ == "__main__":
    main()  # pragma: no cover

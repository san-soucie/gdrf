"""
General utils
Adapted from YOLOv5, https://github.com/ultralytics/yolov5/

"""

import collections
import contextlib
import datetime
import glob
import logging
import os
import platform
import random
import re
import signal
import subprocess
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output

import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torch.backends.cudnn as cudnn

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(
    linewidth=320, formatter={"float_kind": "{:11.5g}".format}
)  # format short g, %precision=5
pd.options.display.max_columns = 10
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count(), 8))  # NumExpr max threads

LOGGER = logging.getLogger(__name__)


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg="", suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel SIGALRM if it's scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def methods(instance):
    # Get class/instance methods
    return [
        f
        for f in dir(instance)
        if callable(getattr(instance, f)) and not f.startswith("__")
    ]


def set_logging(verbose=True):
    logging.basicConfig(
        format="%(message)s", level=logging.INFO if verbose else logging.WARN
    )


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir="."):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def user_config_dir(dir="gdrf", env_var="GDRF_CONFIG_DIR"):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {
            "Windows": "AppData/Roaming",
            "Linux": ".config",
            "Darwin": "Library/Application Support",
        }  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), "")  # OS-specific config dir
        path = (
            path if is_writeable(path) else Path("/tmp")
        ) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / "tmp.txt"
        try:
            with open(file, "w"):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except IOError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows


def is_docker():
    # Is environment a Docker container?
    return Path("/workspace").exists()  # or Path('/.dockerenv').exists()


def is_colab():
    # Is environment a Google Colab instance?
    try:
        import google.colab

        return True
    except Exception:
        return False


def is_pip():
    # Is file in a pip package?
    return "site-packages" in Path(__file__).resolve().parts


def is_ascii(s=""):
    # Is string composed of all ASCII (no UTF) characters?
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)


def emojis(str=""):
    # Return platform-dependent emoji-safe version of string
    return (
        str.encode().decode("ascii", "ignore")
        if platform.system() == "Windows"
        else str
    )


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / 1e6
    else:
        return 0.0


def check_online():
    # Check internet connectivity
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


@try_except
def check_git_status():
    # Recommend 'git pull' if code is out of date
    msg = ", for updates see https://github.com/ultralytics/yolov5"
    print(colorstr("github: "), end="")
    assert Path(".git").exists(), "skipping check (not a git repository)" + msg
    assert not is_docker(), "skipping check (Docker image)" + msg
    assert check_online(), "skipping check (offline)" + msg

    cmd = "git fetch && git config --get remote.origin.url"
    url = (
        check_output(cmd, shell=True, timeout=5).decode().strip().rstrip(".git")
    )  # git fetch
    branch = (
        check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()
    )  # checked out
    n = int(
        check_output(f"git rev-list {branch}..origin/master --count", shell=True)
    )  # commits behind
    if n > 0:
        s = f"⚠️ GDRF is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s = f"up to date with {url} ✅"
    print(emojis(s))  # emoji-safe


def check_python(minimum="3.6.2"):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name="Python ")


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    assert (
        result
    ), f"{name}{minimum} required by GDRF, but {name}{current} is currently installed"


@try_except
def check_requirements(requirements="requirements.txt", exclude=(), install=True):
    # # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    # prefix = colorstr('red', 'bold', 'requirements:')
    # check_python()  # check python version
    # if isinstance(requirements, (str, Path)):  # requirements.txt file
    #     file = Path(requirements)
    #     assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
    #     with open(file) as f:
    #         requirements = [
    #             f"{package}{spec}" for package, spec in toml.load(f)['tool']['poetry']['dependencies'].items()
    #             if not (isinstance(spec, dict) and spec['optional'])
    #         ]
    # else:  # list or tuple of packages
    #     requirements = [x for x in requirements if x not in exclude]
    #
    # n = 0  # number of packages updates
    # for r in requirements:
    #     try:
    #         pkg.require(r)
    #     except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
    #         s = f"{prefix} {r} not found and is required by GDRF"
    #         if install:
    #             print(f"{s}, attempting auto-update...")
    #             try:
    #                 assert check_online(), f"'pip install {r}' skipped (offline)"
    #                 print(check_output(f"pip install '{r}'", shell=True).decode())
    #                 n += 1
    #             except Exception as e:
    #                 print(f'{prefix} {e}')
    #         else:
    #             print(f'{s}. Please install and rerun your command.')
    #
    # if n:  # if packages updated
    #     source = file.resolve() if 'file' in locals() else requirements
    #     s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
    #         f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
    #     print(emojis(s))
    pass  # Not currently functional


def check_suffix(file="gdrf.pt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffixes
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            assert (
                Path(f).suffix.lower() in suffix
            ), f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=(".yaml", ".yml")):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=""):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == "":  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = str(Path(file)).replace(":/", "://")  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file)).name.split("?")[
            0
        ]  # '%2F' to '/', split https://url.com/file.txt?auth
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, file)
        assert (
            Path(file).exists() and Path(file).stat().st_size > 0
        ), f"File download failed: {url}"  # check
        return file
    else:  # search
        files = glob.glob("./**/" + file, recursive=True)  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert (
            len(files) == 1
        ), f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f"Downloading {url} to {f}...")
            if curl:
                os.system(
                    f"curl -L '{url}' -o '{f}' --retry 9 -C -"
                )  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in (".zip", ".gz"):
            print(f"Unzipping {f}...")
            if f.suffix == ".zip":
                s = f"unzip -qo {f} -d {dir}"  # unzip -quiet -overwrite
            elif f.suffix == ".gz":
                s = f"tar xfz {f} --directory {f.parent}"  # unzip
            if delete:  # delete zip file after unzip
                s += f" && rm {f}"
            os.system(s)

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def check_dataset(data, sep=","):

    assert isinstance(data, (str, Path)), f"{data} must be a string or Path object"
    return {"train": data}


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def strip_optimizer(
    f="best.pt", s=""
):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device("cpu"))
    for k in "optimizer", "training_results", "wandb_id", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    print(
        f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB"
    )


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f"git -C {path} describe --tags --long --always"
    try:
        return subprocess.check_output(
            s, shell=True, stderr=subprocess.STDOUT
        ).decode()[:-1]
    except subprocess.CalledProcessError:
        return ""  # not a git repository


def intersect_dicts(da: dict, db: dict, exclude: collections.Iterable = ()) -> dict:
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def select_device(device=""):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f"GDRF {git_describe() or date_modified()} torch {torch.__version__} "  # string
    device = (
        str(device).strip().lower().replace("cuda:", "")
    )  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert (
            torch.cuda.is_available()
        ), f"CUDA unavailable, invalid device {device} requested"  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = (
            device.split(",") if device else "0"
        )  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += "CPU\n"

    LOGGER.info(
        s.encode().decode("ascii", "ignore") if platform.system() == "Windows" else s
    )  # emoji-safe
    return torch.device("cuda" if cuda else "cpu")


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, best_fitness=0.0, patience=30):
        self.best_fitness = best_fitness  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float(
            "inf"
        )  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if (
            fitness >= self.best_fitness
        ):  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (
            self.patience - 1
        )  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f"EarlyStopping patience {self.patience} exceeded, stopping training."
            )
        return stop

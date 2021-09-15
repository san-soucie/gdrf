"""
Train a GDRF model on a custom dataset
Usage:
    $ python path/to/train.py --data mvco.csv
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd

import numpy as np
import pyro.optim
import pyro.infer
import pyro.contrib.gp
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

from gdrf.models import GDRF, MultinomialGDRF, SparseGDRF, SparseMultinomialGDRF, GridGDRF, GridMultinomialGDRF
FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from typing import Union, Optional

from utils.general import init_seeds, strip_optimizer, get_latest_run, check_dataset, check_git_status, check_requirements, \
    check_file, check_yaml, check_suffix, set_logging, colorstr, methods, EarlyStopping, increment_path, select_device
from utils.wandblogger import check_wandb_resume
from utils.loggers import Loggers
from utils.callbacks import Callbacks

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

GDRF_MODEL_DICT = {
    'gdrf': GDRF,
    'multinomialgdrf': MultinomialGDRF,
    'sparsegdrf': SparseGDRF,
    'sparsemultinomialgdrf': SparseMultinomialGDRF,
    'gridgdrf': GridGDRF,
    'gridmultinomialgdrf': GridMultinomialGDRF,
}
OPTIMIZER_DICT = {
    'adagradrmsprop': pyro.optim.AdagradRMSProp,
    'clippedadam': pyro.optim.ClippedAdam,
    'dctadam': pyro.optim.DCTAdam,
    'adadelta': pyro.optim.Adadelta,
    'adagrad': pyro.optim.Adagrad,
    'adam': pyro.optim.Adam,
    'adamw': pyro.optim.AdamW,
    'sparseadam': pyro.optim.SparseAdam,
    'adamax': pyro.optim.Adamax,
    'asgd': pyro.optim.ASGD,
    'sgd': pyro.optim.SGD,
    'rprop': pyro.optim.Rprop,
    'rmsprop': pyro.optim.RMSprop
}
OBJECTIVE_DICT = {
    'elbo': pyro.infer.Trace_ELBO,
    'graphelbo': pyro.infer.TraceGraph_ELBO,
    'renyielbo': pyro.infer.RenyiELBO
}
KERNEL_DICT = {
    'rbf': pyro.contrib.gp.kernels.RBF,
    'matern32': pyro.contrib.gp.kernels.Matern32,
    'matern52': pyro.contrib.gp.kernels.Matern52,
    'exponential': pyro.contrib.gp.kernels.Exponential,
    'rationalquadratic': pyro.contrib.gp.kernels.RationalQuadratic,
}

def train(project: str = "GDRF",
          name: str = 'train',
          device: str = 'cpu',
          exist_ok: bool = True,
          weights: str = '',
          data: str = '',
          dimensions: int = 1,
          epochs: int = 300,
          resume: Union[str, bool] = False,
          nosave: bool = False,
          model_type: str = 'gdrf',
          model_hyp: Optional[dict] = None,
          kernel_type: str = 'rbf',
          kernel_hyp: Optional[dict] = None,
          optimizer_type: str = 'adam',
          optimizer_hyp: Optional[dict] = None,
          objective_type: str = 'elbo',
          objective_hyp: Optional[dict] = None,
          entity: Optional[str] = None,
          upload_dataset: bool = False,
          save_period: int = -1,
          artifact_alias: str = 'latest',
          patience: int = 100,
          ):
    opt = locals()
    callbacks = Callbacks()
    save_dir = Path(str(increment_path(Path(project) / name, exist_ok=exist_ok)))



    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(model_hyp, str):
        with open(model_hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
    if loggers.wandb:
        data_dict = loggers.wandb.data_dict
        if resume:
            epochs, hyp = epochs, model_hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    cuda = device != 'cpu'
    init_seeds(1)
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path = data_dict['train']

    # Dataset
    dataset = pd.read_csv(filepath_or_buffer=data, index_col=list(range(dimensions)), header=0, parse_dates = True, dtype=int)
    xs = torch.from_numpy(dataset.index.values()).float().to(device)
    ws = torch.from_numpy(dataset.values()).float().to(device)
    min_xs = xs.min(dim=0).values.detach().cpu().numpy().tolist()
    max_xs = xs.max(dim=0).values.detach().cpu().numpy().tolist()
    world = list(zip(min_xs, max_xs))

    # Kernel

    kernel = KERNEL_DICT[kernel_type](**kernel_hyp)
    kernel = kernel.to(device)

    # Model
    model = GDRF_MODEL_DICT[model_type](xs=xs, ws=ws, world=world, kernel=kernel, device=device, **model_hyp)

    # Optimizer
    optimizer = OPTIMIZER_DICT[optimizer_type](**optimizer_hyp)

    # Variational Objective
    objective = OBJECTIVE_DICT[objective_type](**objective_hyp)

    # SVI object

    svi = pyro.infer.SVI(model=model.model, guide=model.guide, optim=optimizer, loss=objective)

    LOGGER.info(f"{colorstr('model:')} {type(model).__name__}")
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}")
    LOGGER.info(f"{colorstr('objective:')} {type(objective).__name__}")
    # Resume
    start_epoch, best_fitness = 0, float('inf')

    callbacks.run('on_pretrain_routine_end')

    # Model parameters

    # Start training
    t0 = time.time()
    stopper = EarlyStopping(patience=patience)
    LOGGER.info(f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    with logging_redirect_tqdm():
        pbar = trange(start_epoch, epochs)
        for epoch in pbar:  # epoch ------------------------------------------------------------------
            model.train()

            loss = svi.step()
            model.eval()
            perplexity = model.perplexity(xs, ws).item()

            pbar.set_description(f"Epoch {epoch}")
            pbar.set_postfix(loss=loss, perplexity=perplexity)
            callbacks.run('on_train_epoch_end', epoch=epoch)
            log_vals = [loss, perplexity, model.kernel_lengthscale, model.kernel_variance]
            fi = perplexity
            if fi < best_fitness:
                best_fitness = fi
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if (not nosave) or final_epoch:
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(model).half(),
                        'optimizer': optimizer.get_state(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)
            if stopper(epoch=epoch, fitness=fi):
                break

            # end epoch ---------------------------------------------------------------------------------------
    LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    for f in last, best:
        if f.exists():
            strip_optimizer(f)  # strip optimizers
    callbacks.run('on_train_end', last, best, None, epoch)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return None


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    set_logging(verbose=opt.verbose)
    print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_git_status()
    check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=[])

    # Resume
    if opt.resume and not check_wandb_resume(opt):  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp)  # check YAMLs
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    device = select_device(opt.device)

    # Train
    train(opt.hyp, opt, device, callbacks)


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

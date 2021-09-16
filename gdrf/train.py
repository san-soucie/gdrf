"""
Train a GDRF model on a custom dataset
Usage:
    $ python path/to/train.py --data mvco.csv
"""

import argparse
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import dill as pickle

import pandas as pd

import pyro.optim
import pyro.infer
import pyro.contrib.gp
import torch
import yaml
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from gdrf.models import GDRF, MultinomialGDRF, SparseGDRF, SparseMultinomialGDRF, GridGDRF, GridMultinomialGDRF

from typing import Union

from gdrf.utils.general import init_seeds, strip_optimizer, get_latest_run, check_dataset, check_git_status, \
    check_requirements, check_file, set_logging, colorstr, methods, EarlyStopping, increment_path, \
    select_device
from gdrf.utils.wandblogger import check_wandb_resume
from gdrf.utils.loggers import Loggers
from gdrf.utils.callbacks import Callbacks

FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())

LOGGER = logging.getLogger(__name__)

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


def train(cfg: Union[str, dict] = 'data/cfg.yaml',
          project: str = "wandb/gdrf",
          name: str = 'train',
          device: str = 'cuda:0',
          exist_ok: bool = True,
          weights: str = '',
          data: str = 'data/data.csv',
          dimensions: int = 1,
          epochs: int = 3000,
          resume: Union[str, bool] = True,
          nosave: bool = False,
          entity: str = None,
          upload_dataset: bool = False,
          save_period: int = -1,
          artifact_alias: str = 'latest',
          patience: int = 100,
          verbose: bool = True
          ):
    """
    Trains a GDRF

    :param Union[str, dict] cfg: Config file or dict for training run with model and training hyperparameters
    :param str project: Project name for training run
    :param str name: Run name for training run
    :param str device: Device to store tensors on during training (e.g. 'cpu' or 'cuda:0')
    :param bool exist_ok: Existing project/name ok, do not increment
    :param str weights: Weights.pt file, if starting from pretrained weights
    :param str data: Data.csv file, with first row as column names and first ``dimensions`` columns as indexes
    :param int dimensions: Number of non-stationary index dimensions (e.g. a time series is 1D, x-y data are 2D, etc.)
    :param int epochs: Number of training epochs
    :param Union[str, bool] resume: Resume most recent training
    :param bool nosave: only save final checkpoint
    :param str entity: W&B entity
    :param bool upload_dataset: Upload dataset to W&B
    :param int save_period: Log model after this many epochs
    :param str artifact_alias: version of dataset artifact to be used
    :param int patience: EarlyStopping Patience (number of epochs without improvement before early stopping)
    :param bool verbose: Verbose
    """
    opt = locals()
    callbacks = Callbacks()
    save_dir = Path(str(increment_path(Path(project) / name, exist_ok=exist_ok)))

    set_logging(verbose=verbose)
    print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in opt.items()))
    check_git_status()
    check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=[])


    # Resume
    if resume and not check_wandb_resume(resume):  # resume an interrupted run
        ckpt = resume if isinstance(resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            assert os.path.isfile(Path(ckpt).parent.parent / 'opt.yaml')
            opt = yaml.safe_load(f)  # replace
        with open(Path(ckpt).parent.parent / 'cfg.yaml') as f:
            assert os.path.isfile(Path(ckpt).parent.parent / 'cfg.yaml')
            cfg = yaml.safe_load(f)  # replace
        opt['cfg'], opt['weights'], opt['resume'] = cfg, ckpt, True
        weights = ckpt
        resume = True
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        data = check_file(data)

    if isinstance(cfg, str):
        with open(cfg) as f:
            cfg = yaml.safe_load(f)  # load hyps dict
    model_type = cfg['model']['type']
    model_hyp = cfg['model']['hyperparameters']
    kernel_type = cfg['kernel']['type']
    kernel_hyp = cfg['kernel']['hyperparameters']
    optimizer_type = cfg['optimizer']['type']
    optimizer_hyp = cfg['optimizer']['hyperparameters']
    objective_type = cfg['objective']['type']
    objective_hyp = cfg['objective']['hyperparameters']
    device = select_device(device)

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    LOGGER.info(colorstr('config: ') + ', '.join(f'{k}={v}' for k, v in cfg.items()))

    # Save run settings
    with open(save_dir / 'cfg.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(opt, f, sort_keys=False)
    data_dict = None

    # Loggers
    loggers = Loggers(save_dir, weights, opt, cfg, LOGGER)  # loggers instance
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
    dataset = pd.read_csv(
        filepath_or_buffer=data,
        index_col=list(range(dimensions)),
        header=0,
        parse_dates=True,
    ).fillna(0).astype(int)
    index = dataset.index.values
    index = index - index.min()
    index = index / index.max()
    xs = torch.from_numpy(index).float().to(device)
    if dimensions == 1:
        xs = xs.unsqueeze(-1)
    ws = torch.from_numpy(dataset.values).int().to(device)
    min_xs = xs.min(dim=0).values.detach().cpu().numpy().tolist()
    max_xs = xs.max(dim=0).values.detach().cpu().numpy().tolist()
    world = list(zip(min_xs, max_xs))
    num_observation_categories = len(dataset.columns)


    # Kernel
    for k, v in kernel_hyp.items():
        if isinstance(v, float):
            kernel_hyp[k] = torch.tensor(v).to(device)
    kernel = KERNEL_DICT[kernel_type](input_dim = dimensions, **kernel_hyp)
    kernel = kernel.to(device)

    # Model
    model = GDRF_MODEL_DICT[model_type](
        xs=xs,
        ws=ws,
        world=world,
        kernel=kernel,
        num_observation_categories=num_observation_categories,
        device=device,
        **model_hyp
    )

    # Optimizer
    optimizer = OPTIMIZER_DICT[optimizer_type](optim_args=optimizer_hyp)

    # Variational Objective
    objective = OBJECTIVE_DICT[objective_type](vectorize_particles=True, **objective_hyp)

    # SVI object

    svi = pyro.infer.SVI(model=model.model, guide=model.guide, optim=optimizer, loss=objective)

    LOGGER.info(f"{colorstr('model:')} {type(model).__name__}")
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}")
    LOGGER.info(f"{colorstr('objective:')} {type(objective).__name__}")
    # Resume
    start_epoch, best_fitness = 0, float('-inf')

    callbacks.run('on_pretrain_routine_end')

    # Model parameters

    # Start training
    t0 = time.time()
    stopper = EarlyStopping(best_fitness=best_fitness, patience=patience)
    LOGGER.info(f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    with logging_redirect_tqdm():
        pbar = trange(start_epoch, epochs)
        for epoch in pbar:  # epoch ------------------------------------------------------------------
            model.train()

            loss = svi.step(xs, ws, subsample=False)
            model.eval()
            perplexity = model.perplexity(xs, ws).item()

            pbar.set_description(f"Epoch {epoch}")
            pbar.set_postfix(loss=loss, perplexity=perplexity)
            callbacks.run('on_train_epoch_end', epoch=epoch)
            log_vals = [loss, perplexity, model.kernel_lengthscale, model.kernel_variance]
            fi = -perplexity
            if fi > best_fitness:
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

if __name__ == "__main__":
    train()

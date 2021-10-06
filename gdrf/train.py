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
from typing import Union

import numpy as np
import pandas as pd
import pyro.contrib.gp
import pyro.infer
import pyro.optim
import torch
import yaml
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from gdrf.models import GDRF, MultinomialGDRF, SparseGDRF, SparseMultinomialGDRF
from gdrf.utils.callbacks import Callbacks
from gdrf.utils.general import (
    EarlyStopping,
    check_dataset,
    check_file,
    check_git_status,
    check_requirements,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    methods,
    select_device,
    set_logging,
    strip_optimizer,
)
from gdrf.utils.loggers import Loggers
from gdrf.utils.wandblogger import check_wandb_resume

try:
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())

LOGGER = logging.getLogger(__name__)

GDRF_MODEL_DICT = {
    "gdrf": GDRF,
    "multinomialgdrf": MultinomialGDRF,
    "sparsegdrf": SparseGDRF,
    "sparsemultinomialgdrf": SparseMultinomialGDRF,
}
OPTIMIZER_DICT = {
    "adagradrmsprop": pyro.optim.AdagradRMSProp,
    "clippedadam": pyro.optim.ClippedAdam,
    "dctadam": pyro.optim.DCTAdam,
    "adadelta": pyro.optim.Adadelta,
    "adagrad": pyro.optim.Adagrad,
    "adam": pyro.optim.Adam,
    "adamw": pyro.optim.AdamW,
    "sparseadam": pyro.optim.SparseAdam,
    "adamax": pyro.optim.Adamax,
    "asgd": pyro.optim.ASGD,
    "sgd": pyro.optim.SGD,
    "rprop": pyro.optim.Rprop,
    "rmsprop": pyro.optim.RMSprop,
}
OBJECTIVE_DICT = {
    "elbo": pyro.infer.Trace_ELBO,
    "graphelbo": pyro.infer.TraceGraph_ELBO,
    "renyielbo": pyro.infer.RenyiELBO,
}
KERNEL_DICT = {
    "rbf": pyro.contrib.gp.kernels.RBF,
    "matern32": pyro.contrib.gp.kernels.Matern32,
    "matern52": pyro.contrib.gp.kernels.Matern52,
    "exponential": pyro.contrib.gp.kernels.Exponential,
    "rationalquadratic": pyro.contrib.gp.kernels.RationalQuadratic,
}


def train(  # noqa: C901
    project: str = "wandb/mvco",
    name: str = "mvco_adamax_grid",
    device: str = "cuda:0",
    exist_ok: bool = True,
    weights: str = "",
    data: str = "data/data.csv",
    dimensions: int = 1,
    epochs: int = 3000,
    model_type: str = "sparsemultinomialgdrf",
    num_topics: int = 1,
    dirichlet_param: float = 0.01,
    num_inducing_points: int = 25,
    fixed_inducing_points: bool = True,
    inducing_initialization_method: str = "random",
    jitter: float = 1e-8,
    max_jitter: int = 15,
    kernel_type: str = "rbf",
    kernel_lengthscale: float = 0.1,
    kernel_variance: float = 25.0,
    optimizer_type: str = "adamw",
    optimizer_lr: float = 0.001,
    objective_type: str = "graphelbo",
    objective_num_particles: int = 1,
    objective_renyi_alpha: float = 2.0,
    resume: Union[str, bool] = False,
    nosave: bool = False,
    entity: str = None,
    upload_dataset: bool = False,
    save_period: int = -1,
    artifact_alias: str = "latest",
    patience: int = 100,
    verbose: bool = True,
):
    """
    Trains a GDRF

    :param str project: Project name for training run
    :param str name: Run name for training run
    :param str device: Device to store tensors on during training (e.g. 'cpu' or 'cuda:0')
    :param bool exist_ok: Existing project/name ok, do not increment
    :param str weights: Weights.pt file, if starting from pretrained weights
    :param str data: Data.csv file, with first row as column names and first ``dimensions`` columns as indexes
    :param int dimensions: Number of non-stationary index dimensions (e.g. a time series is 1D, x-y data are 2D, etc.)
    :param int epochs: Number of training epochs
    :param str model_type: 'gdrf', 'multinomialgdrf', 'sparsegdrf', or 'sparsemultinomialgdrf'
    :param int num_topics: Number of GDRF topics
    :param float dirichlet_param: GDRF word-topic dirichlet parameter
    :param int num_inducing_points: Number of inducing points for sparse GDRF.
    :param bool fixed_inducing_points: Whether or not sparse inducing point locations are learned
    :param str inducing_initialization_method: 'random' or 'grid'
    :param float jitter: Diagonal jitter initial value for cholesky decomposition of kernel
    :param int max_jitter: Number of orders of magnitude to try scaling the diagonal jitter before failure
    :param str kernel_type: 'rbf', 'matern32', 'matern52', 'exponential', or 'rationalquadratic'
    :param float kernel_lengthscale: Length scale of kernel, from 0 to 1
    :param float kernel_variance: Overall scaling (variance) of kernel.
    :param str optimizer_type: "adagradrmsprop", "clippedadam", "dctadam", "adadelta", "adagrad", "adam", "adamw", "sparseadam", "adamax", "asgd", "sgd", "rprop", "rmsprop"
    :param float optimizer_lr: Learning rate for optimizer
    :param str objective_type: 'elbo', 'graphelbo', or 'renyielbo'
    :param int objective_num_particles: Number of particles (i.e. latent samples) in ELBO calculations
    :param float objective_renyi_alpha: Renyi alpha. Only valid for renyi objective type
    :param Union[str,bool] resume: Resume most recent training
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
    data_dict = None

    loggers = Loggers(save_dir, weights, opt, LOGGER)  # loggers instance
    if loggers.wandb:
        data_dict = loggers.wandb.data_dict
        opt = wandb.config
        (
            project,
            name,
            device,
            exist_ok,
            weights,
            data,
            dimensions,
            epochs,
            num_topics,
            dirichlet_param,
            num_inducing_points,
            fixed_inducing_points,
            inducing_initialization_method,
            jitter,
            max_jitter,
            kernel_type,
            kernel_lengthscale,
            kernel_variance,
            optimizer_type,
            optimizer_lr,
            objective_type,
            objective_num_particles,
            objective_renyi_alpha,
            resume,
            nosave,
            entity,
            upload_dataset,
            save_period,
            artifact_alias,
            patience,
            verbose,
        ) = (
            opt.project,
            opt.name,
            opt.device,
            opt.exist_ok,
            opt.weights,
            opt.data,
            opt.dimensions,
            opt.epochs,
            opt.num_topics,
            opt.dirichlet_param,
            opt.num_inducing_points,
            opt.fixed_inducing_points,
            opt.inducing_initialization_method,
            opt.jitter,
            opt.max_jitter,
            opt.kernel_type,
            opt.kernel_lengthscale,
            opt.kernel_variance,
            opt.optimizer_type,
            opt.optimizer_lr,
            opt.objective_type,
            opt.objective_num_particles,
            opt.objective_renyi_alpha,
            opt.resume,
            opt.nosave,
            opt.entity,
            opt.upload_dataset,
            opt.save_period,
            opt.artifact_alias,
            opt.patience,
            opt.verbose,
        )

        # Register actions
    for k in methods(loggers):
        callbacks.register_action(k, callback=getattr(loggers, k))

    set_logging(verbose=verbose)
    print(colorstr("train: ") + ", ".join(f"{k}={v}" for k, v in opt.items()))
    check_git_status()
    check_requirements(requirements=FILE.parent / "requirements.txt", exclude=[])

    # Resume
    if resume and not check_wandb_resume(resume):  # resume an interrupted run
        ckpt = (
            resume if isinstance(resume, str) else get_latest_run()
        )  # specified or most recent path
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        with open(Path(ckpt).parent.parent / "opt.yaml") as f:
            assert os.path.isfile(Path(ckpt).parent.parent / "opt.yaml")
            opt = yaml.safe_load(f)  # replace
        opt["weights"], opt["resume"] = ckpt, True
        weights = ckpt
        resume = True
        LOGGER.info(f"Resuming training from {ckpt}")
    else:
        data = check_file(data)

    device = select_device(device)

    pretrained = weights.endswith(".pt")

    # Directories
    w = save_dir / "weights"  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    LOGGER.info(colorstr("config: ") + ", ".join(f"{k}={v}" for k, v in opt.items()))

    # Save run settings
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.safe_dump(opt.as_dict(), f, sort_keys=False)
    data_dict = None

    # Loggers

    # Config
    cuda = device != "cpu"
    init_seeds(1)
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path = data_dict["train"]

    # Dataset
    dataset = (
        pd.read_csv(
            filepath_or_buffer=data,
            index_col=list(range(dimensions)),
            header=0,
            parse_dates=True,
        )
        .fillna(0)
        .astype(int)
    )
    index = dataset.index
    index = index.values if dimensions == 1 else np.array(index.to_list())
    index = index - index.min(axis=-dimensions, keepdims=True)
    index = index / index.max(axis=-dimensions, keepdims=True)
    xs = torch.from_numpy(index).float().to(device)
    if dimensions == 1:
        xs = xs.unsqueeze(-1)
    ws = torch.from_numpy(dataset.values).int().to(device)
    min_xs = xs.min(dim=0).values.detach().cpu().numpy().tolist()
    max_xs = xs.max(dim=0).values.detach().cpu().numpy().tolist()
    world = list(zip(min_xs, max_xs))
    num_observation_categories = len(dataset.columns)

    # Kernel
    kernel_lengthscale = torch.tensor(kernel_lengthscale).to(device)
    kernel_variance = torch.tensor(kernel_variance).to(device)

    kernel = KERNEL_DICT[kernel_type](
        input_dim=dimensions, lengthscale=kernel_lengthscale, variance=kernel_variance
    )
    kernel = kernel.to(device)

    # Model
    model = GDRF_MODEL_DICT[model_type](
        xs=xs,
        ws=ws,
        world=world,
        kernel=kernel,
        num_observation_categories=num_observation_categories,
        device=device,
        num_topic_categories=num_topics,
        dirichlet_param=dirichlet_param,
        n_points=num_inducing_points,
        fixed_inducing_points=fixed_inducing_points,
        inducing_init=inducing_initialization_method,
        maxjitter=max_jitter,
        jitter=jitter,
    )

    # Optimizer

    optimizer = OPTIMIZER_DICT[optimizer_type]({"lr": optimizer_lr})

    # Variational Objective
    objective_hyperparameters = {"num_particles": objective_num_particles}
    if objective_type == "renyielbo":
        objective_hyperparameters["alpha"] = objective_renyi_alpha
    objective = OBJECTIVE_DICT[objective_type](
        vectorize_particles=True, **objective_hyperparameters
    )
    start_epoch, best_fitness = 0, float("-inf")

    if pretrained:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        exclude = []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(
            f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}"
        )
        if ckpt["optimizer"] is not None:
            optimizer.set_state(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if resume:
            assert (
                start_epoch > 0
            ), f"{weights} training to {epochs} epochs is finished, nothing to resume."
        if epochs < start_epoch:
            LOGGER.info(
                f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs."
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, csd
    # SVI object
    scale = pyro.poutine.scale(scale=1.0 / len(xs))
    svi = pyro.infer.SVI(
        model=scale(model.model),
        guide=scale(model.guide),
        optim=optimizer,
        loss=objective,
    )

    LOGGER.info(f"{colorstr('model:')} {type(model).__name__}")
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}")
    LOGGER.info(f"{colorstr('objective:')} {type(objective).__name__}")
    # Resume

    callbacks.run("on_pretrain_routine_end")

    # Model parameters

    # Start training
    t0 = time.time()
    stopper = EarlyStopping(best_fitness=best_fitness, patience=patience)
    LOGGER.info(
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    with logging_redirect_tqdm():
        pbar = trange(start_epoch, epochs, initial=start_epoch, total=epochs)
        for (
            epoch
        ) in (
            pbar
        ):  # epoch ------------------------------------------------------------------
            model.train()

            loss = svi.step(xs, ws, subsample=False)
            model.eval()
            perplexity = model.perplexity(xs, ws).item()

            pbar.set_description(f"Epoch {epoch+1}")
            pbar.set_postfix(loss=loss, perplexity=perplexity)
            callbacks.run("on_train_epoch_end", epoch=epoch)
            log_vals = [
                loss,
                perplexity,
                model.kernel_lengthscale,
                model.kernel_variance,
            ]
            fi = -perplexity
            if fi > best_fitness:
                best_fitness = fi
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if (not nosave) or final_epoch:
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(model).half(),
                    "optimizer": optimizer.get_state(),
                    "wandb_id": loggers.wandb.wandb_run.id if loggers.wandb else None,
                }
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.run(
                    "on_model_save", last, epoch, final_epoch, best_fitness, fi
                )
            if stopper(epoch=epoch - start_epoch, fitness=fi):
                break

            # end epoch ---------------------------------------------------------------------------------------
    LOGGER.info(
        f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours."
    )
    for f in last, best:
        if f.exists():
            strip_optimizer(f)  # strip optimizers
    callbacks.run(
        "on_train_end", last, best, xs, ws, dataset.index, dataset.columns, epoch
    )
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return None


if __name__ == "__main__":
    train()

import matplotlib

from collections import defaultdict

import torch
import pyro
import pyro.optim as optim
import pyro.nn as nn

import pyro.nn.module as module
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.infer as infer
import pyro.infer.autoguide as autoguide
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from tqdm import trange
import pyro.contrib.gp.kernels as kernel
import sys
import numpy as np

from .gdrf import AbstractGDRF

import wandb

@nn.pyro_method
def train_gdrf_offline(xs, ws, optimizer: optim.PyroOptim, objective: infer.ELBO, gdrf: AbstractGDRF, num_steps=100, disable_pbar=False,
                early_stop=True, log=True):
    model = gdrf.model
    guide = gdrf.guide
    # guide = infer.autoguide.AutoDelta(model)
    svi = infer.SVI(model, guide, optimizer, objective)

    losses = []
    log_losses = []
    pbar = trange(num_steps, disable=disable_pbar)
    for idx in pbar:
        loss = svi.step(xs, ws)

        losses.append(loss)
        if log:
            wandb.log({"loss": loss, **gdrf.artifacts(xs, ws)})
        if early_stop:
            log_losses.append(np.log(losses[-1]))
            running_log_loss_mean = np.mean(log_losses[-100:])
            recent_log_loss_resid = log_losses[-100:] - running_log_loss_mean
            loss_criterion = np.max(np.abs(recent_log_loss_resid)) / running_log_loss_mean
            if idx > 100 and loss_criterion < 1e-4:
                print("Reached training convergence")
                break

        pbar.set_description(f"Loss: {losses[-1]:10.10f}")
    return losses

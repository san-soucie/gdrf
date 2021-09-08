import matplotlib

from collections import defaultdict
from enum import Enum
from typing import Optional
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

class TrainingMode(Enum):
    OFFLINE = 1
    ONLINE = 2
    STREAMING = 3

def train_gdrf(xs: torch.Tensor,
               ws: torch.Tensor,
               optimizer: optim.PyroOptim,
               objective: infer.ELBO,
               gdrf: AbstractGDRF,
               num_steps: Optional[int] = 100,
               mode: TrainingMode = TrainingMode.OFFLINE,
               disable_pbar=False,
               early_stop=True,
               log=True,
               log_every=100):
    model = gdrf.model
    guide = gdrf.guide
    # guide = infer.autoguide.AutoDelta(model)
    svi = infer.SVI(model, guide, optimizer, objective)

    losses = []
    log_losses = []
    if mode is not TrainingMode.OFFLINE:
        num_steps = xs.size(-2) # ignore num_steps for streaming/online training
    pbar = trange(num_steps, disable=disable_pbar)
    for idx in pbar:
        training_xs = xs
        training_ws = ws
        if mode is TrainingMode.ONLINE:
            training_xs = training_xs[..., :idx, :]
            training_ws = training_ws[..., :idx, :]
        elif mode is TrainingMode.STREAMING:
            training_xs = training_xs[..., idx, :]
            training_ws = training_ws[..., idx, :]
        loss = svi.step(training_xs, training_ws)

        losses.append(loss)
        if log:
            wandb.log({"loss": loss, **gdrf.artifacts(xs, ws, all=(idx % log_every == 0))})
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

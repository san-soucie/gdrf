
from enum import Enum
from typing import Optional, Collection

import pandas as pd
import torch
import pyro.optim as optim

import pyro.infer as infer

from tqdm import trange
import numpy as np

from .gdrf import AbstractGDRF
from .visualize import categorical_stackplot, matrix_plot

import wandb

class TrainingMode(Enum):
    OFFLINE = 1
    ONLINE = 2
    STREAMING = 3
    GIRDHAR = 4

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
               log_every=100,
               log_every_big=100,
               plot_index: Optional[Collection] = None):
    model = gdrf.model
    guide = gdrf.guide
    # guide = infer.autoguide.AutoDelta(model)
    svi = infer.SVI(model, guide, optimizer, objective)

    losses = []
    log_losses = []
    pbar = trange(num_steps if mode is TrainingMode.OFFLINE else xs.size(-2), disable=disable_pbar)
    for idx in pbar:
        training_xs = xs
        training_ws = ws
        num_substeps = 1
        if mode is TrainingMode.ONLINE or mode is TrainingMode.GIRDHAR:
            training_xs = training_xs[..., :idx+1, :]
            training_ws = training_ws[..., :idx+1, :]
            num_substeps = num_steps
        elif mode is TrainingMode.STREAMING:
            training_xs = training_xs[..., idx, :]
            training_ws = training_ws[..., idx, :]
            num_substeps = num_steps
        for _ in range(num_substeps):
            loss = svi.step(training_xs, training_ws, subsample=mode is TrainingMode.GIRDHAR)

        losses.append(loss)
        if log and idx % log_every == 0:
                artifacts = gdrf.artifacts(xs, ws, all=True)
                artifacts['loss'] = loss
                artifacts['epoch'] = idx
                if idx % (log_every * log_every_big) == 0:
                    artifacts['word_topic_plot'] = wandb.Image(
                        matrix_plot(np.log10(gdrf.word_topic_matrix.detach().cpu().numpy()+1e-15), title="Log Word-topic matrix")
                    )
                    artifacts['perplexity'] = gdrf.perplexity(xs, ws).item(),
                    if gdrf.dims == 1:
                        index = xs if plot_index is None else plot_index
                        plot_xs = xs
                        # if len(index) > 100:
                        #     index = index[::len(index) // 100]
                        #     plot_xs = xs[::len(xs) // 100]
                        df = pd.DataFrame(
                            gdrf.topic_probs(plot_xs).detach().cpu().numpy(),
                            index=index,
                            columns=[f"Topic {i+1}" for i in range(gdrf.K)]
                        )
                        artifacts['topic_plot'] = (wandb.Image(
                            categorical_stackplot(df, title="Topic Probability", label='topic')
                        ))

                wandb.log(artifacts)
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

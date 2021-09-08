import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.nn as nn
import pyro.contrib.gp.kernels as kernel
import pyro.distributions as dist
import os
import numpy as np
import wandb
from gdrf.models import GridMultinomialGDRF, GDRF, train_gdrf, TrainingMode
import pyro.infer as infer
import pyro.optim as optim

from collections import defaultdict
from tqdm import trange
from gdrf.plot_mvco import make_plots, make_stackplot, make_wt_plot, read_the_csv, GROUND_TRUTH_FILE

def run_mvco(K = 5,
             l0 = 0.327,
             sigma=0.889,
             NXu=150,
             beta=9.29,
             num_steps=2000,
             lr=np.exp(-13.046),
             num_particles=10,
             use_cuda=True,
             seed=5,
             prefix='mvco',
             plot=True,
             jitter=1e-6,
             maxjitter=3):
    folder = f"{prefix}_{K}_{l0}_{sigma}_{NXu}_{beta}_{num_steps}_{lr}_{num_particles}_{seed}"

    config = {'number of topics': K,
              'initial kernel lengthscale': l0,
              'initial kernel variance': sigma,
              'number of inducing points': NXu,
              'number of epochs': num_steps,
              'learning rate': lr,
              'number of particles': num_particles,
              'cuda': use_cuda,
              'random seed': seed,
              'output folder': folder,
              'dataset': 'count_by_class_time_seriesCNN_hourly19Aug2021.csv'}





    full_folder = os.path.join('..', 'data', folder)
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    data = pd.read_csv("../data/count_by_class_time_seriesCNN_hourly19Aug2021.csv")

    datatype_dict = {}
    datatype_dict[data.columns[0]] = 'datetime64'
    datatype_dict[data.columns[1]] = 'float64'
    for c in data.columns[2:]:
        datatype_dict[c] = 'int'
    data = data.fillna(0.).astype(datatype_dict)
    bad_cols = {
            'amoeba',
            'bead',
            'bubble',
            'camera_spot',
            'ciliate',
            'coccolithophorid',
            'cryptophyta',
            'detritus',
            'detritus_clear',
            'fecal_pellet',
            'fiber',
            'fiber_TAG_external_detritus',
            'flagellate',
            'flagellate_morphotype1',
            'flagellate_morphotype3',
            'mix',
            'pennate',
            'pennate_Pseudo-nitzschia',
            'pennate_Thalassionema',
            'pennate_morphotype1',
            'pollen',
            'shellfish_larvae',
            'square_unknown',
            'unknown2',
            'zooplankton',
        }
    good_columns = [
        c for c in data.columns if c not in bad_cols
    ]
    data = data[good_columns]

    min_t = min(data[data.columns[0]])
    max_t = max(data[data.columns[0]])
    ts = data[data.columns[0]]
    ws = data[data.columns[2:]].to_numpy(dtype='int')
    ts = ts.to_numpy(dtype=np.float32)
    min_t = min(ts)
    max_t = max(ts)
    ts = torch.Tensor(ts).unsqueeze(-1)
    # n_ts_plot = 100
    # dt = (max_t - min_t) / (n_ts_plot - 1)
    # ts_plot_index = np.arange(min_t, max_t, step = dt)
    # ts_plot = torch.arange(0., 1. + 1 / (n_ts_plot - 1), 1/(n_ts_plot - 1)).unsqueeze(-1)
    ws = torch.Tensor(ws)

    # gt_fig, gt_ax = plt.subplots()
    # gt_fig.set_size_inches(10.5, 3.5)
    # gt_df = read_the_csv(GROUND_TRUTH_FILE, start_col=2, bad_cols=bad_cols)
    # make_stackplot(gt_df, gt_fig, gt_ax, title=f"Ground Truth Taxa")
    # wandb.run.summary['Ground Truth Taxon Distribution'] = wandb.Image(gt_fig)
    # plt.close(gt_fig)

    pyro.set_rng_seed(seed)
    pyro.get_param_store().clear()

    V = ws.shape[1]
    k = kernel.RBF(1, lengthscale=torch.tensor(float(l0), device='cuda' if use_cuda else 'cpu'), variance=torch.tensor(float(sigma), device='cuda' if use_cuda else 'cpu'))
    k.lengthscale = nn.PyroParam(torch.tensor(float(l0), device='cuda' if use_cuda else 'cpu'), constraint=dist.constraints.positive)
    k.variance = nn.PyroParam(torch.tensor(float(sigma), device='cuda' if use_cuda else 'cpu'), constraint=dist.constraints.positive)
    if use_cuda:
        k = k.cuda()
    bounds = [(min_t, max_t)]
    b = torch.tensor(beta, device='cuda' if use_cuda else 'cpu')
    gdrf_model = GridMultinomialGDRF(
        dirichlet_param=b,
        num_topic_categories=K,
        num_observation_categories=V,
        kernel=k,
        n_points=NXu,
        world=bounds,
        device='cuda' if use_cuda else 'cpu',
        whiten=False,
        jitter=jitter,
        maxjitter=maxjitter)
    # wandb.init(config=config)
    # wandb.define_metric('loss', summary='min')
    # wandb.define_metric('perplexity', summary='min')
    losses = train_gdrf(
        xs=ts,
        ws=ws,
        optimizer = optim.AdamW({"lr": lr}),
        objective = infer.Trace_ELBO(num_particles=num_particles,max_plate_nesting=1,vectorize_particles=True),
        gdrf=gdrf_model,
        num_steps=3000,
        mode = TrainingMode.OFFLINE,
        log=False
    )

    #
    #
    # topic_probs = gdrf.topic_probs(ts)
    # word_probs = gdrf.word_probs(ts)
    # word_topic_matrix = gdrf.word_topic_matrix
    # word_topic_matrix_df = pd.DataFrame(word_topic_matrix, index=[f"topic_{k}" for k in range(K)], columns = data.columns[2:])
    # topic_prob_df = pd.DataFrame(topic_probs, index=data[data.columns[0]], columns=[f"topic_{k}" for k in range(1,K+1)])
    # word_prob_df = pd.DataFrame(word_probs, index=data[data.columns[0]], columns=data.columns[2:])
    # topic_prob_fn = os.path.join(full_folder, '_'.join([folder, 'topic_probs.csv']))
    # word_prob_fn = os.path.join(full_folder, '_'.join([folder, 'word_probs.csv']))
    # word_topic_matrix_fn = os.path.join(full_folder, '_'.join([folder, 'word_topic_matrix.csv']))
    # if not os.path.exists(full_folder):
    #     os.mkdir(full_folder)
    # topic_prob_df.to_csv(topic_prob_fn)
    # word_prob_df.to_csv(word_prob_fn)
    # word_topic_matrix_df.to_csv(word_topic_matrix_fn)
    # if plot:
    #     make_plots(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_mvco(K = 6,
             l0 = 0.1,
             sigma=0.1,
             NXu=150,
             beta=np.exp(-5.8692987),
             num_steps=25000,
             lr=0.001,
             num_particles=1,
             use_cuda=True,
             seed=57575,
             prefix='aug_24_21_mvco',
             plot=False,
             jitter=1e-10,
             maxjitter=20)

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

def run_mvco(num_topics = 5,
             initial_lengthscale = 0.327,
             initial_variance=0.889,
             num_inducing=150,
             dirichlet_param=9.29,
             num_steps=2000,
             learning_rate=np.exp(-13.046),
             num_particles=10,
             device='cpu',
             seed=5,
             training_name='mvco',
             training_project='mvco',
             jitter=1e-6,
             maxjitter=3,
             early_stop=False):

    if device[:4] == 'cuda':
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
    ts = data[data.columns[0]]
    min_t = min(ts)
    max_t = max(ts)
    ts_norm = (ts - min_t) / (max_t - min_t)
    ws = data[data.columns[2:]].to_numpy(dtype='int')
    ts = ts.to_numpy(dtype=np.float32)
    ts /= 1e18
    min_t = min(ts)
    max_t = max(ts)
    xs = torch.Tensor(ts_norm).unsqueeze(-1)
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
    k = kernel.RBF(1, lengthscale=torch.tensor(float(initial_lengthscale), device=device), variance=torch.tensor(float(initial_variance), device=device))
    k.lengthscale = nn.PyroParam(torch.tensor(float(initial_lengthscale), device=device), constraint=dist.constraints.positive)
    k.variance = nn.PyroParam(torch.tensor(float(initial_variance), device=device), constraint=dist.constraints.positive)
    if device[:4] == 'cuda':
        k = k.cuda()
    bounds = [(0.0, 1.0)]
    b = torch.tensor(dirichlet_param, device=device)
    gdrf_model = GridMultinomialGDRF(
        dirichlet_param=b,
        num_topic_categories=num_topics,
        num_observation_categories=V,
        kernel=k,
        n_points=num_inducing,
        world=bounds,
        device=device,
        whiten=False,
        jitter=jitter,
        maxjitter=maxjitter)
    config = {'number of topics': num_topics,
              'initial kernel lengthscale': initial_lengthscale,
              'initial kernel variance': initial_variance,
              'number of inducing points': num_inducing,
              'number of epochs': num_steps,
              'learning rate': learning_rate,
              'number of particles': num_particles,
              'device': device,
              'random seed': seed,
              'dataset': 'count_by_class_time_seriesCNN_hourly19Aug2021.csv'}
    wandb.init(project=training_project, name=training_name, config=config)
    wandb.define_metric('loss', summary='min')
    wandb.define_metric('perplexity', summary='min')
    losses = train_gdrf(
        xs=xs,
        ws=ws,
        optimizer = optim.ClippedAdam({"lr": learning_rate, 'betas': (0.95, 0.999)}),
        objective = infer.Trace_ELBO(num_particles=num_particles,max_plate_nesting=1,vectorize_particles=True),
        gdrf=gdrf_model,
        num_steps=num_steps,
        mode = TrainingMode.ONLINE,
        log=True,
        log_every=50,
        plot_index=data[data.columns[0]],
        early_stop=early_stop
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
    parser.add_argument('-K', '--num-topics', default=6, type=int,
                        help='Number of topics to use')
    parser.add_argument('-l', '--initial-lengthscale', default=0.1, type=float,
                        help='Initial kernel length scale')
    parser.add_argument('-s', '--initial-variance', default=50.0, type=float,
                        help='Initial kernel variance')
    parser.add_argument('-i', '--num-inducing', default=500, type=int,
                        help='Number of inducing points')
    parser.add_argument('-b', '--dirichlet-param', default=1.e-8, type=float,
                        help='Single-value Dirichlet hyperparameter for word-topic matrix')
    parser.add_argument('-n', '--num-steps', default=10000, type=int,
                        help='Number of training steps')
    parser.add_argument('-L', '--learning-rate', default=1.e-4, type=float,
                        help='Learning rate hyperparamater for pytorch optimizer')
    parser.add_argument('-p', '--num-particles', default=5, type=int,
                        help='Number of variational inference posterior samples per training step')
    parser.add_argument('-D', '--device', default='cuda',
                        help='Device to train on')
    parser.add_argument('-S', '--seed', default=57575, type=int,
                        help='Random seed')
    parser.add_argument('-N', '--training-name', default='mvco',
                        help='Name of training run')
    parser.add_argument('-P', '--training-project', default='sparse_multinomial_gdrf_1000inducing',
                        help='Name of training run')
    parser.add_argument('-j', '--jitter', default=1e-10, type=float,
                        help='Initial diagonal jitter to use for calculating cholesky decompositions')
    parser.add_argument('-J', '--maxjitter', default=20, type=float,
                        help='Maximum jitter multiplier exponent')
    parser.add_argument('--early-stop', action='store_true',
                        help='Stop training once convergence is reached')

    args = parser.parse_args()
    run_mvco(**vars(args))

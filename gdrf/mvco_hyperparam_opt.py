import pandas as pd
from gdrf2 import SparseMultinomialGDRF
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.nn as nn
import pyro.distributions as dist
import pyro.contrib.gp as gp
import os
import skopt
import sklearn
import numpy as np
from histogramdd import histogramdd
from typing import Optional
from collections import defaultdict

from utils import generate_data_2d_circles
from mvco import run_mvco

kernels = [
    'matern32',
    'matern52',
]

kernel_factory_dict = {
    'rbf': gp.kernels.RBF,
    'matern32': gp.kernels.Matern32,
    'matern52': gp.kernels.Matern52,
    'rq': gp.kernels.RationalQuadratic,
    'exponential': gp.kernels.Exponential,
    'cosine': gp.kernels.Cosine,
    'periodic': gp.kernels.Periodic
}
def kernel_factory(name: str, d: int, l: float, v: float, device: str = 'cuda'):
    return kernel_factory_dict[name](
        input_dim=d,
        lengthscale=torch.tensor(l).to(device),
        variance=torch.tensor(v).to(device))


def _inner_mutual_info(p_xy: torch.Tensor, p_x: torch.Tensor, p_y: torch.Tensor, x_dims: int):
    assert p_xy.is_sparse
    summands = [
        p_xy[idx] * torch.log(p_xy[idx] / (p_x[idx[:x_dims]] * p_y[idx[x_dims:]])) for idx in
        zip(*p_xy.indices().detach().cpu().numpy())
    ]
    return sum(x for x in summands if not x.isnan().item())

def mutual_information(x: np.array, y: np.array, normalize=True):
    if len(x.shape) == 1:
        x = np.array(x)[:, np.newaxis]
    if len(y.shape) == 1:
        y = np.array(y)[:, np.newaxis]
    x_dims = x.shape[-1]
    y_dims = y.shape[-1]
    xy = np.concatenate([x, y], axis=-1)
    p_xy, _ = histogramdd(xy)
    p_xy = p_xy.coalesce()

    p_xy /= torch.sparse.sum(p_xy)
    p_x = torch.sparse.sum(p_xy, dim=tuple(range(-1, -1-y_dims, -1)))
    p_y = torch.sparse.sum(p_xy, dim=tuple(range(x_dims)))
    mutual_info = _inner_mutual_info(p_xy, p_x, p_y, x_dims)
    if normalize and mutual_info.is_nonzero():
        h_x = -torch.sum(p_x._values() * torch.log(p_x._values()))
        h_y = -torch.sum(p_y._values() * torch.log(p_y._values()))
        mutual_info /= torch.sqrt(h_x * h_y)
    if mutual_info.detach().cpu().isnan():
        print(mutual_info)
    return mutual_info.detach().cpu().item()

def perplexity(probs: np.ndarray, index: np.ndarray):
    return np.exp((np.log(probs) * index).sum() / index.sum())

def _hp_opt_objective(xs: torch.Tensor,
                      ws: torch.Tensor,
                      log_beta_0: float,
                      nXu: int,
                      lengthscale: float,
                      variance: float,
                      log_lr: float,
                      K: int,
                      V: int,
                      n_dims: int,
                      device: str,
                      num_particles: int,
                      num_metaparticles: int,
                      num_steps: int,
                      index_xs: torch.Tensor,
                      index: np.ndarray,
                      verbose: bool,
                      metric: str = 'mi',
                      jitter=1e-10,
                      maxjitter=10,
                      ):
    if verbose:
        print(
            f"--------\nHyperparameters:\n\nlog_beta={log_beta_0}\nNXu={nXu}\nl={lengthscale}\nv={variance}\nk={K}\nlr={np.exp(log_lr)}\n--------")
    lr = np.exp(log_lr)
    beta_0 = np.exp(log_beta_0)
    metric_vals = []
    for _ in range(num_metaparticles):
        k = kernel_factory('exponential', d=n_dims, l=lengthscale, v=variance, device=device)
        if device == 'cuda':
            k = k.cuda()
        bounds = [(0.0, 1.0) for _ in range(n_dims)]
        Xu = torch.rand(nXu, n_dims, device=device)
        b = torch.tensor(beta_0, device=device)
        pyro.clear_param_store()
        gdrf = SparseMultinomialGDRF(b=b, k=K, v=V, s=k, Xu=Xu, n=xs.shape[0], world=bounds, cuda=(device == 'cuda'),
                                     jitter = jitter, whiten=False, maxjitter=maxjitter)
        gdrf.train_model(x=xs, w=ws, num_steps=num_steps, lr=lr, num_particles=num_particles,
                         early_stop=False, disable_pbar=(not verbose), log=False)
        if metric == 'mi' or metric == 'mi_normalized':
            summary = gdrf.log_topic_probs(index_xs).detach().cpu().T.numpy()
            metric_func = lambda p, i: -mutual_information(p,i, normalize=(metric == 'mi_normalized'))
        elif metric == 'perplexity':
            summary = gdrf.word_probs(index_xs)
            metric_func = lambda p, i: perplexity(p, i)
        else:
            raise ValueError('metric must be "mi", "mi_normalized", or "perplexity')
        del gdrf, b, Xu, k
        metric_vals.append(metric_func(summary, index))
        print(f'{metric}: {metric_vals[-1]}')
    return min(metric_vals)

def hyperparameter_optimize(
    xs: torch.Tensor,
    ws: torch.Tensor,
    V: int,
    index: np.ndarray,
    index_xs: torch.Tensor,
    num_steps: int = 500,
    num_particles: int = 1,
    num_metaparticles: int = 1,
    n_calls: int = 50,
    device: str = 'cuda',
    fixed_topics: Optional[int] = None,
    verbose: bool = True,
    metric: str = 'mi'
):
    n_dims = xs.shape[1]


    hyperparameters = [
        skopt.space.Real(name='log_beta_0', low=-6., high=1.),
        skopt.space.Integer(name='nXu', low=5, high=50),
        #  skopt.space.Categorical(name='kernel', categories=kernels),
        skopt.space.Real(name='lengthscale', low=1e-4, high=1.0),
        skopt.space.Real(name='variance', low=1e-4, high=1.0),
        skopt.space.Real(name='log_lr', low=np.log(1e-10), high=np.log(1e-2)),
    ]

    if fixed_topics is None:
        hyperparameters.append(skopt.space.Integer(name='K', low=2, high=25))

        @skopt.utils.use_named_args(dimensions=hyperparameters)
        def hp_opt_objective(log_beta_0, nXu, lengthscale, variance, log_lr, K):
            return _hp_opt_objective(xs=xs, ws=ws, log_beta_0=log_beta_0, nXu=nXu, lengthscale=lengthscale, variance=variance,
                                     log_lr=log_lr, K=K, V=V, n_dims=n_dims, device=device, num_particles=num_particles,
                                     num_metaparticles=num_metaparticles, num_steps=num_steps, index_xs=index_xs, index=index,
                                     verbose=verbose, metric=metric)
    else:
        K = fixed_topics
        @skopt.utils.use_named_args(dimensions=hyperparameters)
        def hp_opt_objective(log_beta_0, nXu, lengthscale, variance, log_lr):
            return _hp_opt_objective(xs=xs, ws=ws, log_beta_0=log_beta_0, nXu=nXu, lengthscale=lengthscale, variance=variance,
                                     log_lr=log_lr, K=K, V=V, n_dims=n_dims, device=device, num_particles=num_particles,
                                     num_metaparticles=num_metaparticles, num_steps=num_steps, index_xs=index_xs, index=index,
                                     verbose=verbose, metric=metric)


    result = skopt.gp_minimize(
        func=hp_opt_objective,
        dimensions=hyperparameters,
        acq_func="gp_hedge",
        n_calls=n_calls,
        verbose=verbose
    )

    if verbose:

        print(f"Best Fitness: {result.fun}")
        print(f"Best parameters: {result.x}")

    return result

def hyperparameter_optimize_just_beta(
    xs: torch.Tensor,
    ws: torch.Tensor,
    V: int,
    index: np.ndarray,
    index_xs: torch.Tensor,
    num_steps: int = 500,
    num_particles: int = 1,
    num_metaparticles: int = 1,
    n_calls: int = 50,
    device: str = 'cuda',
    fixed_topics: Optional[int] = None,
    verbose: bool = True,
    metric: str = 'mi',
    nXu: int = 50,
    lengthscale: float = 1e-2,
    variance: float = 1e-1,
    lr: float = 1e-3,
    K: int = 6,
    jitter=1e-10,
    maxjitter=10,
):
    n_dims = xs.shape[1]


    hyperparameters = [

        skopt.space.Integer(name='nXu', low=5, high=50),
        #  skopt.space.Categorical(name='kernel', categories=kernels),
        skopt.space.Real(name='lengthscale', low=1e-4, high=1.0),
        skopt.space.Real(name='variance', low=1e-4, high=1.0),
        skopt.space.Real(name='log_lr', low=np.log(1e-10), high=np.log(1e-2)),
    ]

    if fixed_topics is None:
        hyperparameters.append(skopt.space.Integer(name='K', low=2, high=25))

    def hp_opt_objective(log_beta_0):
        return _hp_opt_objective(xs=xs, ws=ws, log_beta_0=log_beta_0[0], nXu=nXu, lengthscale=lengthscale, variance=variance,
                                 log_lr=np.log(lr), K=K, V=V, n_dims=n_dims, device=device, num_particles=num_particles,
                                 num_metaparticles=num_metaparticles, num_steps=num_steps, index_xs=index_xs, index=index,
                                 verbose=verbose, metric=metric, jitter=jitter, maxjitter=maxjitter)


    result = skopt.gp_minimize(
        func=hp_opt_objective,
        dimensions=[(-10., 2.)],
        acq_func="gp_hedge",
        n_calls=n_calls,
        verbose=verbose
    )

    if verbose:

        print(f"Best Fitness: {result.fun}")
        print(f"Best parameters: {result.x}")
        betas = np.array([np.exp(r[0]) for r in result.x_iters])
        vals = result.func_vals
        idx = np.argsort(betas)
        plt.semilogx(betas[idx], vals[idx])
        plt.title("Perplexity versus Dirichlet Parameter")
        plt.show()

    return result

def main_mvco(use_cuda = True):
    device = 'cuda' if use_cuda else 'cpu'
    data = pd.read_csv("../data/count_by_class_time_seriesCNN_hourly19Aug2021.csv")

    datatype_dict = {}
    datatype_dict[data.columns[0]] = 'datetime64'
    datatype_dict[data.columns[1]] = 'float64'
    for c in data.columns[2:]:
        datatype_dict[c] = 'int'
    data = data.fillna(0.).astype(datatype_dict)
    good_columns = [
        c for c in data.columns if c not in {
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
    ]
    data = data[good_columns]

    min_t = min(data[data.columns[0]])
    max_t = max(data[data.columns[0]])
    ts = (data[data.columns[0]] - min_t) / (max_t - min_t)
    ws = data[data.columns[2:]].to_numpy(dtype='int')
    ts = torch.Tensor(ts).unsqueeze(-1).to(device)
    ws = torch.Tensor(ws).to(device)
    v = len(data.columns[2:])
    index = data[data.columns[0]].apply(lambda x: x.month // 4)

    result = hyperparameter_optimize(
        xs=ts,
        ws=ws,
        V=v,
        index_xs=ts,
        index=ws.detach().cpu().numpy(),
        device=device,
        metric='perplexity'
    )

    hyps = {
        n: v for v, n in zip(
            result.x, [
                'beta',
                'NXu',
                'l0',
                'sigma',
                'lr',
                'K',
            ]
        )
    }
    for s in ['lr', 'beta']:
        hyps[s] = np.exp(hyps[s])

    run_mvco(**hyps,
             num_steps=10000,
             num_particles=3,
             use_cuda=True,
             seed=5**4 - 13,
             prefix='perplexity_bayes_mvco',
             jitter=1e-10,
             maxjitter=10)


def main_mvco_just_beta(
    datafile: str = "../data/count_by_class_time_seriesCNN_hourly19Aug2021.csv",
    nXu: int = 150,
    lengthscale: float = 0.1,
    variance: float = 0.1,
    lr: float = 0.001,
    K: int = 6,
    use_cuda = True,
    num_steps=10000,
    num_particles=3,
    seed=5**4 - 13,
    prefix='perplexity_bayes_mvco',
    jitter=1e-10,
    maxjitter=10,
    num_calls: int = 50
):
    device = 'cuda' if use_cuda else 'cpu'
    data = pd.read_csv(datafile)

    datatype_dict = {}
    datatype_dict[data.columns[0]] = 'datetime64'
    datatype_dict[data.columns[1]] = 'float64'
    for c in data.columns[2:]:
        datatype_dict[c] = 'int'
    data = data.fillna(0.).astype(datatype_dict)
    good_columns = [
        c for c in data.columns if c not in {
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
    ]
    data = data[good_columns]

    min_t = min(data[data.columns[0]])
    max_t = max(data[data.columns[0]])
    ts = (data[data.columns[0]] - min_t) / (max_t - min_t)
    ws = data[data.columns[2:]].to_numpy(dtype='int')
    ts = torch.Tensor(ts).unsqueeze(-1).to(device)
    ws = torch.Tensor(ws).to(device)
    v = len(data.columns[2:])
    index = data[data.columns[0]].apply(lambda x: x.month // 4)


    result = hyperparameter_optimize_just_beta(
        xs=ts,
        ws=ws,
        V=v,
        index_xs=ts,
        index=ws.detach().cpu().numpy(),
        num_steps=1000,
        device=device,
        metric='perplexity',
        K=K,
        lr=lr,
        lengthscale=lengthscale,
        variance=variance,
        nXu = nXu,
        verbose=True,
        jitter=jitter,
        maxjitter=maxjitter,
        n_calls=num_calls
    )

    hyps = {
        n: v for v, n in zip(
            result.x, [
                'beta',
                'NXu',
                'l0',
                'sigma',
                'lr',
                'K',
            ]
        )
    }
    for s in ['lr', 'beta']:
        hyps[s] = np.exp(hyps[s])
    hyps = {
        'beta': np.exp(result.x[0]),
        'NXu': nXu,
        'l0': lengthscale,
        'sigma': variance,
        'lr': lr,
        'K': K
    }

    run_mvco(**hyps,
             num_steps=num_steps,
             num_particles=num_particles,
             use_cuda=use_cuda,
             seed=seed,
             prefix=prefix,
             jitter=jitter,
             maxjitter=maxjitter)

def main_artificial(use_cuda = True, n_samples = 10, min_k = 2, max_k = 20, filename='test_artificial.csv', verbose=False):
    device = 'cuda' if use_cuda else 'cpu'
    columns = ['n_topics', 'n_vocab', 'n_circle', 'width', 'height', 'radius', 'eta', 'seed', 'dirichlet_param', 'n_inducing', 'lengthscale', 'variance', 'lr', 'mutual_info']
    results = defaultdict(list)
    for K in range(min_k, max_k+1):
        for seed in range(n_samples):
            random_seed = (2**7 // 13) + seed * 11
            gen = np.random.default_rng(seed = random_seed)
            V = gen.integers(low = K * 5, high = K * 50)
            W = 100
            H = 100
            N = K * 2
            R = 10
            eta = 0.0001
            centers, topics, p_v_z, xs, words = generate_data_2d_circles(K=K, V=V, N=N, W=W, H=H, R=R, eta=eta, seed=seed, device=device, permute=True, constant_background=True)
            index = topics.flatten().detach().cpu().numpy()
            index_xs = xs
            hyps = hyperparameter_optimize(xs=xs, ws=words, V=V, index=index, index_xs=index_xs, num_steps=1000, num_particles=10, num_metaparticles=1, n_calls=50, device=device, fixed_topics=K, verbose=verbose)
            log_beta_0, nXu, lengthscale, variance, log_lr = hyps.x
            mutual_info = -hyps.fun
            lr = np.exp(log_lr)
            beta_0 = np.exp(log_beta_0)
            results['n_topics'].append(K)
            results['n_vocab'].append(V)
            results['n_circle'].append(N)
            results['width'].append(W)
            results['height'].append(H)
            results['radius'].append(R)
            results['eta'].append(eta)
            results['seed'].append(seed)
            results['dirichlet_param'].append(beta_0)
            results['n_inducing'].append(nXu)
            results['lengthscale'].append(lengthscale)
            results['variance'].append(variance)
            results['lr'].append(lr)
            results['mutual_info'].append(mutual_info)
    pd.DataFrame(results).to_csv('../data/' + filename)


if __name__ == "__main__":
    # main_artificial(n_samples=10, max_k=10, verbose=True)
    main_mvco_just_beta(
        datafile = "../data/count_by_class_time_seriesCNN_hourly19Aug2021.csv",
        nXu = 250,
        lengthscale = 0.001,
        variance = 10.,
        lr= 0.01,
        K = 5,
        use_cuda = True,
        num_steps=25000,
        num_particles=1,
        seed=5**4 - 13,
        prefix='perplexity_bayes_mvco',
        jitter=1e-10,
        maxjitter=20,
        num_calls=100
    )

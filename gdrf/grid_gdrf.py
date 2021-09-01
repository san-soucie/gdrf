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
from utils import generate_simple_data, generate_data_2d_circles, jittercholesky, dirichlet_param
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from tqdm import trange
import pyro.contrib.gp.kernels as kernel
import sys
import numpy as np
from typing import Union
import wandb

class SparseGridMultinomialGDRF(gp.Parameterized):
    def __init__(self,
                 b: torch.Tensor,
                 k,
                 v,
                 n,
                 world: list[tuple[float, float]],
                 n_grid_points: Union[list[int], int],
                 m=lambda d: d.sum(dim=-1),
                 s: kernel.Kernel = kernel.Matern52(1),
                 jitter=1e-6,
                 f=lambda m: torch.softmax(m, -2),
                 device='cuda:0',
                 maxjitter: int = 3,
                 noise = None,
                 whiten=True):
        if type(n_grid_points) == int:
            n_grid_points = [n_grid_points for _ in world]
        else:
            assert len(world) == len(n_grid_points), "Number of dimensions must agree for 'world' and 'n_grid_points'"
        super().__init__()
        self.gpu = device[:4] == 'cuda'
        self.device = device
        if self.gpu:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.mean_function = m
        self.kernel = s
        self.K = k
        self.V = v
        self.N = n
        beta = dirichlet_param(b, self.K, self.V, device=device)

        dg = [(b[1] - b[0]) / (n - 1) for b, n in zip(world, n_grid_points)]

        self.Xu = torch.stack([x.flatten() for x in torch.meshgrid([torch.arange(b[0], b[1]+d, step=d) for b, d in zip(world, dg)])], dim=-1).to(device)
        self.beta = beta.to(device)
        phi_map = torch.ones(self.beta.shape, dtype=self.beta.dtype) / self.V

        self.phi_map = nn.PyroParam(phi_map, constraint=dist.constraints.stack([dist.constraints.simplex for _ in range(self.K)], dim=0))
        self.f = f
        self.guide = None
        self.jitter = jitter
        self.maxjitter = maxjitter
        self.latent_shape = torch.Size((k,))
        self.M = self.Xu.size(0)
        self.D = self.Xu.size(1)
        u_loc = torch.zeros((self.K, self.M), dtype=self.Xu.dtype).to(device)
        self.u_loc = torch.nn.Parameter(u_loc)
        # self.u_loc = nn.PyroParam(u_loc)
        identity = dist.util.eye_like(self.Xu, self.M)
        u_scale_tril = identity.repeat((self.K, 1, 1)).to(device)
        self.u_scale_tril = nn.PyroParam(u_scale_tril, dist.constraints.lower_cholesky)
        noise = torch.tensor(1.).to(device) if noise is None else noise
        self.noise = nn.PyroParam(noise, constraint=dist.constraints.positive)
        self.whiten = whiten

    @property
    def kernel_lengthscale(self):
        return pyro.param('kernel.lengthscale').detach().cpu().numpy()

    @property
    def kernel_variance(self):
        return pyro.param('kernel.variance').detach().cpu().numpy()


    @property
    def word_topic_matrix(self):
        return pyro.param('phi_map').detach().cpu().numpy()

    @property
    def word_topic_matrix_tensor(self):
        return pyro.param('phi_map')

    @property
    def inducing_points(self):
        return self.Xu.detach().cpu().numpy()
    @property
    def inducing_points_mean(self):
        return pyro.param('u_loc').detach().cpu().numpy()

    @property
    def inducing_points_scale_tril(self):
        return pyro.param('u_scale_tril').detach().cpu().numpy()

    def log_topic_probs(self, xs):
        posterior_kernel = self.kernel
        posterior_u_loc = self.inducing_points_mean
        posterior_u_scale_tril = self.inducing_points_scale_tril
        Luu = jittercholesky(
            posterior_kernel(torch.Tensor(self.inducing_points)).contiguous(),
            self.M,
            self.jitter,
            self.maxjitter
        )
        f_loc, _ = gp.util.conditional(
            xs,
            torch.Tensor(self.inducing_points),
            posterior_kernel,
            torch.Tensor(posterior_u_loc),
            torch.Tensor(posterior_u_scale_tril),
            Luu,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )
        return f_loc

    def topic_probs(self, xs):
        return self.f(self.log_topic_probs(xs)).detach().cpu().numpy().T

    def topic_probs_tensor(self, xs):
        return self.f(self.log_topic_probs(xs))

    def word_probs(self, xs):
        return self.topic_probs(xs) @ self.word_topic_matrix

    def word_probs_tensor(self, xs):
        return self.topic_probs_tensor(xs).T @ self.word_topic_matrix_tensor

    def ml_topics(self, xs):
        return np.argmax(self.topic_probs(xs), axis=1)

    def ml_words(self, xs):
        return np.argmax(self.word_probs(xs), axis=1)

    def random_words(self, xs, seed=None):
        rng = np.random.default_rng(seed=seed)
        probs = self.word_probs(xs)
        return np.array([rng.choice(a=self.V, p=probs[idx, :]) for idx in range(xs.shape[0])])



    @nn.pyro_method
    def model(self, x, w=None):

        Kuu = self.kernel(self.Xu).contiguous()
        Luu = jittercholesky(Kuu, self.M, self.jitter, self.maxjitter)
        u_scale_tril = dist.util.eye_like(self.Xu, self.M).to(self.device) if self.whiten else Luu
        zero_loc = self.Xu.new_zeros(self.u_loc.shape).to(self.device)

        f_loc, f_var = gp.util.conditional(
            x,
            self.Xu,
            self.kernel,
            self.u_loc,
            self.u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )

        f_loc = f_loc + self.mean_function(x)
        with pyro.plate("topics", self.K, device=self.device) as idx:
            pyro.sample(
                "u",
                dist.MultivariateNormal(zero_loc, scale_tril=u_scale_tril).to_event(
                    zero_loc.dim() - 1
                ),
            )
            mu = pyro.sample("mu", dist.Normal(f_loc, f_var + self.noise).to_event(1))
            phi = pyro.sample("phi", dist.Dirichlet(self.beta.to(self.device)))
        topic_probs = self.f(mu).transpose(-2, -1)
        probs = torch.matmul(topic_probs.to(self.device), phi.to(self.device))
        with pyro.plate("obs", size=w.shape[-2], device=self.device) as idx:
            w = pyro.sample("w", dist.Multinomial(probs=probs[..., idx, :], validate_args=False), obs=w[..., idx, :])
        return w

    def forward(self, x):
        return self.model(x)

    @nn.pyro_method
    def variational_distribution(self, x, w=None):
        kernel = self.kernel
        Xu = self.Xu
        u_loc = self.u_loc
        u_scale_tril = self.u_scale_tril
        Kuu = kernel(Xu).contiguous()
        Luu = jittercholesky(Kuu, self.M, self.jitter, self.maxjitter)
        f_loc, f_var = gp.util.conditional(
            x,
            Xu,
            kernel,
            u_loc,
            u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )
        f_loc = f_loc + self.mean_function(x)
        phi_map = self.phi_map
        with pyro.plate("topics", self.K, device=self.device) as idx:
            pyro.sample(
                "u",
                dist.MultivariateNormal(u_loc, scale_tril=u_scale_tril).to_event(
                    u_loc.dim() - 1
                ),
            )
            pyro.sample("mu", dist.Normal(f_loc, f_var).to_event(1))
            pyro.sample("phi", dist.Delta(phi_map).to_event(1))

    def perplexity(self, x, w):
        return np.exp((w * np.log(self.word_probs(x))).sum() / -w.sum())

    def perplexity_tensor(self, x, w):
        return ((w * self.word_probs_tensor(x).log()).sum() / -w.sum()).exp()


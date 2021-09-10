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
import pyro.contrib.gp as gp
import sys
import numpy as np

from pyro.ops.indexing import Vindex

import wandb

from abc import ABCMeta, abstractmethod
from .topic_model import SpatioTemporalTopicModel, scale_decorator

from typing import Callable, Union, Optional, Any
from .utils import validate_dirichlet_param, jittercholesky
from .abstract_gdrf import AbstractGDRF


class GDRF(AbstractGDRF):

    def __init__(self,
                 num_observation_categories: int,
                 num_topic_categories: int,
                 world: list[tuple[float, float]],
                 kernel: gp.kernels.Kernel,
                 dirichlet_param: Union[float, torch.Tensor],
                 xs: torch.Tensor,
                 ws: Optional[torch.Tensor] = None,
                 mean_function: Callable = lambda x: 0.0,
                 link_function: Callable = lambda x: torch.softmax(x, -2),
                 device: str = 'cpu',
                 whiten: bool = False,
                 jitter: float = 1e-8,
                 maxjitter: int = 5):
        super().__init__(num_observation_categories, num_topic_categories, world, kernel, dirichlet_param,
                         mean_function=mean_function, link_function=link_function, device=device, )
        if ws is not None and ws.size(-2) != xs.size(-2):
            raise ValueError("Expected the number of input data points equal to the number of output data points, "
                             "but got {} and {}", xs.size(-2), ws.size(-2))
        self.xs = xs
        self.ws = ws
        self._word_topic_matrix_map = nn.PyroParam(
            self._dirichlet_param.to(self.device),
            constraint=dist.constraints.stack([dist.constraints.simplex for _ in range(self._K)], dim=-2)
        )
        self._jitter = jitter
        self._maxjitter = maxjitter
        self._whiten = whiten
        N = self.xs.size(-2)
        self.latent_shape = torch.Size([self._K])
        f_loc = self.xs.new_zeros(self.latent_shape + (N, ))
        self.f_loc = nn.PyroParam(f_loc)
        identity = dist.util.eye_like(self.xs, N)
        f_scale_tril = identity.repeat(self.latent_shape + (1, 1))
        self.f_scale_tril = nn.PyroParam(f_scale_tril, dist.constraints.lower_cholesky)
        self._sample_latent = True

    @scale_decorator('xs')
    def artifacts(self, xs: torch.Tensor, ws: torch.Tensor, all: bool = False):
        return {
            'perplexity': self.perplexity(xs, ws).item(),
            'kernel variance': self._get('_kernel.variance'),
            'kernel lengthscale': self._get('_kernel.lengthscale'),
            'topic probabilities': self.topic_probs(xs).detach().cpu().numpy(),
            'word-topic matrix': self.word_topic_matrix.detach().cpu().numpy(),
            'word probabilities': self.word_probs(xs).detach().cpu().numpy(),
        }

    @nn.pyro_method
    @scale_decorator('xs')
    def log_topic_probs(self, xs):
        self._check_Xnew_shape(xs)
        self.set_mode("guide")
        loc, _ = gp.util.conditional(
            xs,
            self.xs,
            self._kernel,
            self.f_loc,
            self.f_scale_tril,
            full_cov=False,
            whiten=self._whiten,
            jitter=self._jitter,
        )
        return loc

    @property
    def word_topic_matrix(self) -> torch.Tensor:
        return self._word_topic_matrix_map

    def _check_Xnew_shape(self, Xnew: torch.Tensor):
        if self._xs_train is None:
            raise RuntimeError("Must train model before evaluating")
        if Xnew.dim() != self._xs_train.dim():
            raise ValueError(
                "Train data and test data should have the same "
                "number of dimensions, but got {} and {}.".format(
                    self._xs_train.dim(), Xnew.dim()
                )
            )
        if self._xs_train.shape[1:] != Xnew.shape[1:]:
            raise ValueError(
                "Train data and test data should have the same "
                "shape of features, but got {} and {}.".format(
                    self._xs_train.shape[1:], Xnew.shape[1:]
                )
            )

    @nn.pyro_method
    @scale_decorator('xs')
    def model(self, xs, ws):
        self.set_mode("model")
        N = xs.size(-2)
        Kff = self._kernel(xs)
        Kff.view(-1)[:: N + 1] += self.jitter + self.noise  # add noise to diagonal
        Lff = jittercholesky(Kff, N, self.jitter, self.maxjitter)

        zero_loc = xs.new_zeros(self.f_loc.shape)
        if self.whiten:
            identity = dist.util.eye_like(xs, N)
            mu = pyro.sample(
                self._pyro_get_fullname("mu"),
                dist.MultivariateNormal(zero_loc, scale_tril=identity).to_event(
                    zero_loc.dim() - 1
                ),
            )
            f_scale_tril = Lff.matmul(self.f_scale_tril)
            f_loc = Lff.matmul(self.f_loc.unsqueeze(-1)).squeeze(-1)
        else:
            mu = pyro.sample(
                self._pyro_get_fullname("mu"),
                dist.MultivariateNormal(zero_loc, scale_tril=Lff).to_event(
                    zero_loc.dim() - 1
                ),
            )
            f_scale_tril = self.f_scale_tril
            f_loc = self.f_loc
        f_loc = f_loc + self.mean_function(xs)
        f_var = f_scale_tril.pow(2).sum(dim=-1)
        f = dist.Normal(f_loc, f_var.sqrt())()
        f_swap = f.transpose(-2, -1)
        f_res = self._link_function(f_swap)
        topic_dist = dist.Categorical(f_res)
        phi = pyro.sample(self._pyro_get_fullname("phi"), dist.Dirichlet(self._dirichlet_param).to_event(zero_loc.dim()-1))

        with pyro.plate("obs", device=self.device):
            z = pyro.sample(self._pyro_get_fullname('z'), dist.Categorical(probs=topic_dist).to_event())
            w = pyro.sample(self._pyro_get_fullname("w"), dist.Categorical(probs=Vindex(phi)[..., z, :]), obs=ws)
        return w

    @nn.pyro_method
    @scale_decorator('xs')
    def guide(self, xs, ws):
        self.set_mode("guide")
        self._load_pyro_samples()
        pyro.sample(
            self._pyro_get_fullname("mu"),
            dist.MultivariateNormal(self.f_loc, scale_tril=self.f_scale_tril).to_event(
                self.f_loc.dim() - 1
            ),
        )
        f_var = self.f_scale_tril.pow(2).sum(dim=-1)
        f = dist.Normal(self.f_loc, f_var.sqrt())()
        f_swap = f.transpose(-2, -1)
        f_res = self._link_function(f_swap)
        topic_dist = dist.Categorical(f_res)
        phi = pyro.sample("phi", dist.Dirichlet(self.beta).to_event(1))

        with pyro.plate("obs", device=self.device):
            z = pyro.sample(self._pyro_get_fullname('z'), dist.Categorical(probs=topic_dist).to_event())

    @nn.pyro_method
    @scale_decorator('Xnew')
    def forward(self, Xnew, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, f_{loc}, f_{scale\_tril})
            = \mathcal{N}(loc, cov).

        .. note:: Variational parameters ``f_loc``, ``f_scale_tril``, together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        loc, cov = gp.util.conditional(
            Xnew,
            self.scale(self.xs),
            self._kernel,
            self.f_loc,
            self.f_scale_tril,
            full_cov=full_cov,
            whiten=self._whiten,
            jitter=self._jitter,
        )
        return loc + self._mean_function(Xnew), cov


class MultinomialGDRF(GDRF):
    @nn.pyro_method
    @scale_decorator('xs')
    def model(self, xs, ws):
        self.set_mode("model")

        N = xs.size(-2)
        Kff = self._kernel(xs)
        Kff.view(-1)[:: N + 1] += self.jitter + self.noise  # add noise to diagonal
        Lff = jittercholesky(Kff, N, self.jitter, self.maxjitter)

        zero_loc = xs.new_zeros(self.f_loc.shape)
        if self.whiten:
            identity = dist.util.eye_like(xs, N)
            mu = pyro.sample(
                self._pyro_get_fullname("mu"),
                dist.MultivariateNormal(zero_loc, scale_tril=identity).to_event(
                    zero_loc.dim() - 1
                ),
            )
            f_scale_tril = Lff.matmul(self.f_scale_tril)
            f_loc = Lff.matmul(self.f_loc.unsqueeze(-1)).squeeze(-1)
        else:
            mu = pyro.sample(
                self._pyro_get_fullname("mu"),
                dist.MultivariateNormal(zero_loc, scale_tril=Lff).to_event(
                    zero_loc.dim() - 1
                ),
            )
            f_scale_tril = self.f_scale_tril
            f_loc = self.f_loc
        f_loc = f_loc + self.mean_function(xs)
        f_var = f_scale_tril.pow(2).sum(dim=-1)
        f = dist.Normal(f_loc, f_var.sqrt())()
        f_swap = f.transpose(-2, -1)
        f_res = self._link_function(f_swap)
        phi = pyro.sample(self._pyro_get_fullname("phi"), dist.Dirichlet(self.beta).to_event(zero_loc.dim()-1))
        word_dist = f_res @ phi
        with pyro.plate("obs", device=self.device):
            w = pyro.sample(self._pyro_get_fullname("w"), dist.Multinomial(probs=word_dist), obs=ws, validate_args=False)
        return w

    @nn.pyro_method
    @scale_decorator('xs')
    def guide(self, xs, ws):
        self.set_mode("guide")
        self._load_pyro_samples()
        pyro.sample(
            self._pyro_get_fullname("mu"),
            dist.MultivariateNormal(self.f_loc, scale_tril=self.f_scale_tril).to_event(
                self.f_loc.dim() - 1
            ),
        )
        f_var = self.f_scale_tril.pow(2).sum(dim=-1)
        f = dist.Normal(self.f_loc, f_var.sqrt())()
        f_swap = f.transpose(-2, -1)
        f_res = self._link_function(f_swap)
        topic_dist = dist.Categorical(f_res)
        phi = pyro.sample(self._pyro_get_fullname("phi"), dist.Dirichlet(self.beta).to_event(1))


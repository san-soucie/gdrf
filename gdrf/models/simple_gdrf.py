from typing import Callable, List, Optional, Tuple, Union

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.nn as nn
import torch
from pyro.ops.indexing import Vindex

from .abstract_gdrf import AbstractGDRF
from .topic_model import scale_decorator
from .utils import jittercholesky


class SimpleGDRF(AbstractGDRF):
    def __init__(
        self,
        num_observation_categories: int,
        num_topic_categories: int,
        world: List[Tuple[float, float]],
        kernel: gp.kernels.Kernel,
        dirichlet_param: Union[float, torch.Tensor],
        xs: torch.Tensor,
        ws: Optional[torch.Tensor] = None,
        mean_function: Callable = None,
        link_function: Callable = None,
        device: str = "cpu",
        jitter: float = 1e-8,
        maxjitter: int = 5,
        randomize_wt_matrix: bool = False,
        randomize_metric: Optional[
            Callable[[torch.Tensor, AbstractGDRF], float]
        ] = None,
        randomize_iters: int = 100,
        noise: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_observation_categories,
            num_topic_categories,
            world,
            kernel,
            dirichlet_param,
            mean_function=mean_function,
            link_function=link_function,
            device=device,
        )
        if ws is not None and ws.size(-2) != xs.size(-2):
            raise ValueError(
                "Expected the number of input data points equal to the number of output data points, "
                "but got {} and {}",
                xs.size(-2),
                ws.size(-2),
            )
        if link_function is not None:
            raise ValueError("link_function must be None")

        self.xs = xs
        self.ws = ws

        self._jitter = jitter
        self._maxjitter = maxjitter
        N = self.xs.size(-2)
        self.latent_shape = torch.Size([self._K])
        f_loc = self.xs.new_zeros(self.latent_shape + (N,))
        self.f_loc = nn.PyroParam(f_loc)
        f_scale_tril = dist.util.eye_like(self.xs, N).repeat(self.latent_shape + (1, 1))
        self.f_scale_tril = nn.PyroParam(f_scale_tril, dist.constraints.lower_cholesky)
        self._sample_latent = True

        self._word_topic_matrix_map = self.make_wt_matrix(
            self._dirichlet_param,
            self._K,
            self.device,
            randomize=randomize_wt_matrix,
            randomize_metric=randomize_metric,
            randomize_iters=randomize_iters,
        )

        noise = torch.tensor(1.0) if noise is None else noise
        self._noise = nn.PyroParam(noise, constraint=dist.constraints.positive)
        # self._initialize_params(metric=lambda: self.perplexity(xs, ws), extrema=min)

    @scale_decorator("xs")
    def artifacts(self, xs: torch.Tensor, ws: torch.Tensor, all: bool = False):
        return {
            "perplexity": self.perplexity(xs, ws).item(),
            "kernel variance": self._get("_kernel.variance"),
            "kernel lengthscale": self._get("_kernel.lengthscale"),
            "topic probabilities": self.topic_probs(xs).detach().cpu().numpy(),
            "word-topic matrix": self.word_topic_matrix.detach().cpu().numpy(),
            "word probabilities": self.word_probs(xs).detach().cpu().numpy(),
        }

    @nn.pyro_method
    @scale_decorator("xs")
    def log_topic_probs(self, xs):
        self._check_Xnew_shape(xs)
        self.set_mode("guide")
        Lff = jittercholesky(
            self._kernel(xs).contiguous(),
            xs.size(0),
            self._jitter,
            self._maxjitter,
        )
        loc, _ = gp.util.conditional(
            xs,
            self.xs,
            self._kernel,
            self.f_loc,
            self.f_scale_tril,
            Lff=Lff,
            full_cov=False,
            whiten=self._whiten,
            jitter=self._jitter,
        )
        return loc

    @property
    def word_topic_matrix(self) -> torch.Tensor:
        return self._word_topic_matrix_map

    @nn.pyro_method
    def topic_probs(self, xs):
        return torch.softmax(self.log_topic_probs(xs), -2).T

    def _check_Xnew_shape(self, Xnew: torch.Tensor):
        if self.xs is None:
            raise RuntimeError("Must train model before evaluating")
        if Xnew.dim() != self.xs.dim():
            raise ValueError(
                "Train data and test data should have the same "
                "number of dimensions, but got {} and {}.".format(
                    self.xs.dim(), Xnew.dim()
                )
            )
        if self.xs.shape[1:] != Xnew.shape[1:]:
            raise ValueError(
                "Train data and test data should have the same "
                "shape of features, but got {} and {}.".format(
                    self.xs.shape[1:], Xnew.shape[1:]
                )
            )

    @nn.pyro_method
    @scale_decorator("xs")
    def model(self, xs, ws, subsample=False):
        self.set_mode("model")
        N = xs.size(-2)
        covariance = torch.block_diag(*[self._kernel(xs) for _ in range(self._K)])
        scale_tril = jittercholesky(
            covariance, N * self._K, self._jitter, self._maxjitter, self._noise
        )
        mean = torch.cat(tuple(self._mean_function(xs) for _ in range(self._K)), dim=-1)
        log_p_z_given_x_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(mean, scale_tril=scale_tril),
            transforms=[
                dist.transforms.ReshapeTransform(mean.size(), torch.Size((N, self._K))),
            ],
        )

        p_w_given_z_dist = dist.Dirichlet(self._dirichlet_param).expand(
            torch.Size((self._K,))
        )

        p_z_given_x = torch.softmax(
            pyro.sample(self._pyro_get_fullname("p_z_given_x"), log_p_z_given_x_dist),
            -2,
        )
        p_w_given_z = pyro.sample(
            self._pyro_get_fullname("p_w_given_z"), p_w_given_z_dist
        )

        with pyro.plate("obs", ws.size(-2), device=self.device):
            z = pyro.sample(self._pyro_get_fullname("z"), dist.Categorical(p_z_given_x))
            w = pyro.sample(
                self._pyro_get_fullname("w"),
                dist.Categorical(probs=Vindex(p_w_given_z)[..., z, :]),
                obs=ws,
            )
        return w

    @nn.pyro_method
    @scale_decorator("xs")
    def guide(self, xs, ws, subsample=False):
        self.set_mode("guide")
        self._load_pyro_samples()

        N = xs.size(-2)
        mean = torch.cat(tuple(self.f_loc for _ in range(self._K)), dim=-1)
        log_p_z_given_x_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(self.f_loc, scale_tril=self.f_scale_tril),
            transforms=[
                dist.transforms.ReshapeTransform(mean.size(), torch.Size((N, self._K))),
            ],
        )

        p_w_given_z_dist = dist.Delta(self._word_topic_matrix_map).to_event(1)

        p_z_given_x = torch.softmax(
            pyro.sample(self._pyro_get_fullname("p_z_given_x"), log_p_z_given_x_dist),
            -2,
        )
        pyro.sample(self._pyro_get_fullname("p_w_given_z"), p_w_given_z_dist)

        with pyro.plate("obs", ws.size(-2), device=self.device):
            pyro.sample(self._pyro_get_fullname("z"), dist.Categorical(p_z_given_x))

    @nn.pyro_method
    @scale_decorator("Xnew")
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


class SimpleMultinomialGDRF(SimpleGDRF):
    @nn.pyro_method
    @scale_decorator("xs")
    def model(self, xs, ws, subsample=False):
        self.set_mode("model")
        N = xs.size(-2)
        covariance = torch.block_diag(*[self._kernel(xs) for _ in range(self._K)])
        scale_tril = jittercholesky(
            covariance, N * self._K, self._jitter, self._maxjitter, self._noise
        )
        mean = torch.cat(tuple(self._mean_function(xs) for _ in range(self._K)), dim=-1)
        log_p_z_given_x_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(mean, scale_tril=scale_tril),
            transforms=[
                dist.transforms.ReshapeTransform(mean.size(), torch.Size((N, self._K))),
            ],
        )

        p_w_given_z_dist = dist.Dirichlet(self._dirichlet_param).expand(
            torch.Size((self._K,))
        )

        p_z_given_x = torch.softmax(
            pyro.sample(self._pyro_get_fullname("p_z_given_x"), log_p_z_given_x_dist),
            -2,
        )
        p_w_given_z = pyro.sample(
            self._pyro_get_fullname("p_w_given_z"), p_w_given_z_dist
        )
        with pyro.plate("obs", N, device=self.device):
            w = pyro.sample(
                self._pyro_get_fullname("w"),
                dist.Multinomial(probs=p_z_given_x @ p_w_given_z, validate_args=False),
                obs=ws,
            )
        return w

    @nn.pyro_method
    @scale_decorator("xs")
    def guide(self, xs, ws, subsample=False):
        self.set_mode("guide")
        self._load_pyro_samples()
        N = xs.size(-2)
        log_p_z_given_x_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(self.f_loc, scale_tril=self.f_scale_tril),
            transforms=[
                dist.transforms.ReshapeTransform(
                    self.f_loc.size(), torch.Size((N, self._K))
                ),
            ],
        )

        p_w_given_z_dist = dist.Delta(self._word_topic_matrix_map).to_event(1)

        torch.softmax(
            pyro.sample(self._pyro_get_fullname("p_z_given_x"), log_p_z_given_x_dist),
            -2,
        )
        pyro.sample(self._pyro_get_fullname("p_w_given_z"), p_w_given_z_dist)

from typing import Callable, Optional, Union

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.nn as nn
import torch
from pyro.ops.indexing import Vindex

from .abstract_gdrf import AbstractGDRF
from .topic_model import scale_decorator
from .utils import jittercholesky


class SparseGDRF(AbstractGDRF):
    def __init__(
        self,
        num_observation_categories: int,
        num_topic_categories: int,
        world: list[tuple[float, float]],
        kernel: gp.kernels.Kernel,
        dirichlet_param: Union[float, torch.Tensor],
        n_points: Union[int, list[int]],
        fixed_inducing_points: bool = False,
        inducing_init: str = "random",
        mean_function: Callable = None,
        link_function: Callable = None,
        noise: Optional[float] = None,
        device: str = "cpu",
        whiten: bool = False,
        jitter: float = 1e-8,
        maxjitter: int = 5,
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
        self._n_points = (
            [n_points for _ in world] if isinstance(n_points, int) else n_points
        )
        self._fixed_inducing_points = fixed_inducing_points
        if inducing_init == "random":
            points = [
                torch.sort(torch.rand(self._n_points[i]))[0].to(device)
                * self._delta_bounds[i]
                + self._lower_bounds[i]
                for i in range(self.dims)
            ]
        elif inducing_init == "grid":
            points = [
                torch.arange(
                    b[0],
                    b[1] + (b[1] - b[0]) / (n - 1) - 1e-10,
                    (b[1] - b[0]) / (n - 1),
                )
                for b, n in zip(world, self._n_points)
            ]
        else:
            raise ValueError(
                f"inducing_init argument {inducing_init} not valid. Only 'random' and 'grid' are currently supported"
            )
        inducing_points = torch.stack(
            [x.flatten() for x in torch.meshgrid(points)]
        ).T.to(device)
        scaled_inducing_points = self.scale(inducing_points.to(device))
        scaled_world = [(0.0, 1.0) for _ in world]
        self._inducing_points = (
            scaled_inducing_points
            if fixed_inducing_points
            else nn.PyroParam(
                scaled_inducing_points,
                constraint=dist.constraints.stack(
                    [dist.constraints.interval(*c) for c in scaled_world], dim=-1
                ),
            )
        )
        self._word_topic_matrix_map = nn.PyroParam(
            self._dirichlet_param.to(self.device),
            constraint=dist.constraints.stack(
                [dist.constraints.simplex for _ in range(self._K)], dim=-2
            ),
        )
        self._jitter = jitter
        self._maxjitter = maxjitter
        self._whiten = whiten
        self.latent_shape = torch.Size([self._K])
        self.M = self._inducing_points.size(-2)
        self.D = self._inducing_points.size(-1)
        u_loc = torch.zeros((self.K, self.M), dtype=self._inducing_points.dtype).to(
            device
        )
        self.u_loc = torch.nn.Parameter(u_loc)
        identity = dist.util.eye_like(self._inducing_points, self.M)
        u_scale_tril = identity.repeat((self.K, 1, 1)).float()
        self.u_scale_tril = nn.PyroParam(u_scale_tril, dist.constraints.lower_cholesky)
        noise = torch.tensor(1.0) if noise is None else noise
        self.noise = nn.PyroParam(noise, constraint=dist.constraints.positive)
        self._sample_latent = True

    def _check_Xnew_shape(self, Xnew: torch.Tensor):
        if Xnew.dim() != self._inducing_points.dim():
            raise ValueError(
                "Inducing points and test data should have the same "
                "number of dimensions, but got {} and {}.".format(
                    self._inducing_points.dim(), Xnew.dim()
                )
            )
        if self._inducing_points.shape[1:] != Xnew.shape[1:]:
            raise ValueError(
                "Inducing points and test data should have the same "
                "shape of features, but got {} and {}.".format(
                    self._inducing_points.shape[1:], Xnew.shape[1:]
                )
            )

    @scale_decorator("xs")
    def artifacts(self, xs: torch.Tensor, ws: torch.Tensor, all: bool = False):
        ret = {
            # 'perplexity': self.perplexity(xs, ws).item(),
            "kernel variance": self._get("_kernel.variance"),
            "kernel lengthscale": self._get("_kernel.lengthscale"),
            # 'topic probabilities': self.topic_probs(xs).detach().cpu().numpy(),
            # 'word-topic matrix': self.word_topic_matrix.detach().cpu().numpy(),
            # 'word probabilities': self.word_probs(xs).detach().cpu().numpy(),
        }
        if not self._fixed_inducing_points:
            ret["inducing_points"] = self._inducing_points.detach().cpu().numpy()
        return ret

    @nn.pyro_method
    @scale_decorator("xs")
    def log_topic_probs(self, xs):
        self._check_Xnew_shape(xs)
        self.set_mode("guide")
        posterior_kernel = self._kernel
        posterior_u_loc = self.u_loc
        posterior_u_scale_tril = self.u_scale_tril
        Luu = jittercholesky(
            posterior_kernel(self._inducing_points).contiguous(),
            self.M,
            self._jitter,
            self._maxjitter,
        )
        f_loc, _ = gp.util.conditional(
            xs,
            self._inducing_points,
            posterior_kernel,
            posterior_u_loc,
            posterior_u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self._whiten,
            jitter=self._jitter,
        )
        return f_loc

    @property
    def word_topic_matrix(self) -> torch.Tensor:
        return self._word_topic_matrix_map

    @nn.pyro_method
    @scale_decorator("xs")
    def model(self, xs, ws, subsample=False):
        self.set_mode("model")

        Kuu = self._kernel(self._inducing_points).contiguous()
        Luu = jittercholesky(Kuu, self.M, self._jitter, self._maxjitter)
        u_scale_tril = (
            dist.util.eye_like(self._inducing_points, self.M) if self._whiten else Luu
        )
        zero_loc = self._inducing_points.new_zeros(self.u_loc.shape)

        f_loc, f_var = gp.util.conditional(
            xs,
            self._inducing_points,
            self._kernel,
            self.u_loc,
            self.u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self._whiten,
            jitter=self._jitter,
        )

        f_loc = f_loc + self._mean_function(xs)
        with pyro.plate("topics", self._K, device=self.device) as idx:
            pyro.sample(
                self._pyro_get_fullname("u"),
                dist.MultivariateNormal(zero_loc, scale_tril=u_scale_tril).to_event(
                    zero_loc.dim() - 1
                ),
            )
            mu = pyro.sample(
                self._pyro_get_fullname("mu"),
                dist.Normal(f_loc, f_var + self.noise).to_event(1),
            )
            phi = pyro.sample(
                self._pyro_get_fullname("phi"), dist.Dirichlet(self._dirichlet_param)
            )
        with pyro.plate("obs", ws.size(-2), device=self.device):
            z = pyro.sample(
                self._pyro_get_fullname("z"), dist.Categorical(self._link_function(mu))
            )
            w = pyro.sample("w", dist.Categorical(probs=Vindex(phi)[..., z, :]), obs=ws)
        return w

    @nn.pyro_method
    @scale_decorator("xs")
    def guide(self, xs, ws, subsample=False):
        self.set_mode("guide")
        self._load_pyro_samples()

        kernel = self._kernel
        Xu = self._inducing_points
        u_loc = self.u_loc
        u_scale_tril = self.u_scale_tril
        Kuu = kernel(Xu).contiguous()
        Luu = jittercholesky(Kuu, self.M, self._jitter, self._maxjitter)
        f_loc, f_var = gp.util.conditional(
            xs,
            Xu,
            kernel,
            u_loc,
            u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self._whiten,
            jitter=self._jitter,
        )
        f_loc = f_loc + self._mean_function(xs)
        phi_map = self._word_topic_matrix_map
        with pyro.plate("topics", self.K, device=self.device) as idx:
            pyro.sample(
                self._pyro_get_fullname("u"),
                dist.MultivariateNormal(u_loc, scale_tril=u_scale_tril).to_event(
                    u_loc.dim() - 1
                ),
            )
            mu = pyro.sample(
                self._pyro_get_fullname("mu"), dist.Normal(f_loc, f_var).to_event(1)
            )
            pyro.sample(self._pyro_get_fullname("phi"), dist.Delta(phi_map).to_event(1))

        with pyro.plate("obs", ws.size(-2), device=self.device):
            z = pyro.sample(
                self._pyro_get_fullname("z"), dist.Categorical(self._link_function(mu))
            )

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
        self.set_mode("guide")

        posterior_kernel = self._kernel
        posterior_u_loc = self.u_loc
        posterior_u_scale_tril = self.u_scale_tril
        Luu = jittercholesky(
            posterior_kernel(torch.Tensor(self._inducing_points)).contiguous(),
            self.M,
            self.jitter,
            self.maxjitter,
        )
        f_loc, f_cov = gp.util.conditional(
            Xnew,
            torch.Tensor(self._inducing_points),
            posterior_kernel,
            torch.Tensor(posterior_u_loc),
            torch.Tensor(posterior_u_scale_tril),
            Luu,
            full_cov=full_cov,
            whiten=self._whiten,
            jitter=self._jitter,
        )
        return f_loc + self._mean_function(Xnew), f_cov


class SparseMultinomialGDRF(SparseGDRF):
    @nn.pyro_method
    @scale_decorator("xs")
    def model(self, xs, ws, subsample=False):
        self.set_mode("model")
        Kuu = self._kernel(self._inducing_points).contiguous()
        Luu = jittercholesky(Kuu, self.M, self._jitter, self._maxjitter)
        u_scale_tril = (
            dist.util.eye_like(self._inducing_points, self.M) if self._whiten else Luu
        )
        zero_loc = self._inducing_points.new_zeros(self.u_loc.shape)

        f_loc, f_var = gp.util.conditional(
            xs,
            self._inducing_points,
            self._kernel,
            self.u_loc,
            self.u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self._whiten,
            jitter=self._jitter,
        )

        f_loc = f_loc + self._mean_function(xs)
        with pyro.plate("topics", self._K, device=self.device) as idx:
            pyro.sample(
                self._pyro_get_fullname("u"),
                dist.MultivariateNormal(zero_loc, scale_tril=u_scale_tril).to_event(
                    zero_loc.dim() - 1
                ),
            )
            mu = pyro.sample(
                self._pyro_get_fullname("mu"),
                dist.Normal(f_loc, f_var + self.noise).to_event(1),
            )
            phi = pyro.sample(
                self._pyro_get_fullname("phi"), dist.Dirichlet(self._dirichlet_param)
            )
        topic_probs = self._link_function(mu).transpose(-2, -1)
        probs = torch.matmul(topic_probs, phi)
        with pyro.plate(
            "obs",
            ws.size(-2),
            device=self.device,
        ) as idx:
            w = pyro.sample(
                "w",
                dist.Multinomial(probs=probs[..., idx, :], validate_args=False),
                obs=ws[..., idx, :],
            )
        return w

    @nn.pyro_method
    @scale_decorator("xs")
    def guide(self, xs, ws, subsample=False):
        self.set_mode("guide")
        self._load_pyro_samples()
        xs = self.scale(xs)

        kernel = self._kernel
        Xu = self._inducing_points
        u_loc = self.u_loc
        u_scale_tril = self.u_scale_tril
        Kuu = kernel(Xu).contiguous()
        Luu = jittercholesky(Kuu, self.M, self._jitter, self._maxjitter)
        f_loc, f_var = gp.util.conditional(
            xs,
            Xu,
            kernel,
            u_loc,
            u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self._whiten,
            jitter=self._jitter,
        )
        f_loc = f_loc + self._mean_function(xs)
        phi_map = self._word_topic_matrix_map
        with pyro.plate("topics", self.K, device=self.device) as idx:
            pyro.sample(
                self._pyro_get_fullname("u"),
                dist.MultivariateNormal(u_loc, scale_tril=u_scale_tril).to_event(
                    u_loc.dim() - 1
                ),
            )
            mu = pyro.sample(
                self._pyro_get_fullname("mu"), dist.Normal(f_loc, f_var).to_event(1)
            )
            pyro.sample(self._pyro_get_fullname("phi"), dist.Delta(phi_map).to_event(1))


# class GridGDRF(SparseGDRF):
#     def __init__(self,
#                  num_observation_categories: int,
#                  num_topic_categories: int,
#                  world: list[tuple[float, float]],
#                  kernel: gp.kernels.Kernel,
#                  dirichlet_param: Union[float, torch.Tensor],
#                  n_points: Union[int, list[int]],
#                  mean_function: Callable = None,
#                  link_function: Callable = None,
#                  noise: Optional[float] = None,
#                  device: str = 'cpu',
#                  whiten: bool = False,
#                  jitter: float = 1e-8,
#                  maxjitter: int = 5,
#                  **kwargs):
#         assert isinstance(n_points, int) or (len(world) == len(n_points)) , "single int or list len(world) for n_points"
#         n_points = [n_points for _ in world] if isinstance(n_points, int) else n_points
#         points = [torch.arange(b[0], b[1] + (b[1] - b[0]) / (n - 1) - 1e-10, (b[1] - b[0]) / (n - 1)) for b, n in zip(world, n_points)]
#         inducing_points = torch.stack([x.flatten() for x in torch.meshgrid(points)]).T.to(device)
#         super().__init__(
#             num_observation_categories=num_observation_categories,
#             num_topic_categories=num_topic_categories,
#             world=world,
#             kernel=kernel,
#             dirichlet_param=dirichlet_param,
#             inducing_points=inducing_points,
#             fixed_inducing_points=True,
#             mean_function=mean_function,
#             link_function=link_function,
#             noise=noise,
#             device=device,
#             whiten=whiten,
#             jitter=jitter,
#             maxjitter=maxjitter
#         )
#
#
# class GridMultinomialGDRF(SparseMultinomialGDRF):
#     def __init__(self,
#                  num_observation_categories: int,
#                  num_topic_categories: int,
#                  world: list[tuple[float, float]],
#                  kernel: gp.kernels.Kernel,
#                  dirichlet_param: Union[float, torch.Tensor],
#                  n_points: Union[int, list[int]],
#                  mean_function: Callable = None,
#                  link_function: Callable = None,
#                  noise: Optional[float] = None,
#                  device: str = 'cpu',
#                  whiten: bool = False,
#                  jitter: float = 1e-8,
#                  maxjitter: int = 5,
#                  **kwargs):
#         assert isinstance(n_points, int) or len(world) == len(n_points), "single int or list len(world) for n_points"
#         n_points = [n_points for _ in world] if isinstance(n_points, int) else n_points
#         points = [torch.arange(b[0], b[1] + (b[1] - b[0]) / (n - 1) - 1e-10, (b[1] - b[0]) / (n - 1)) for b, n in zip(world, n_points)]
#         inducing_points = torch.stack([x.flatten() for x in torch.meshgrid(points)]).T.to(device)
#         super().__init__(
#             num_observation_categories=num_observation_categories,
#             num_topic_categories=num_topic_categories,
#             world=world,
#             kernel=kernel,
#             dirichlet_param=dirichlet_param,
#             inducing_points=inducing_points,
#             fixed_inducing_points=True,
#             mean_function=mean_function,
#             link_function=link_function,
#             noise=noise,
#             device=device,
#             whiten=whiten,
#             jitter=jitter,
#             maxjitter=maxjitter
#         )

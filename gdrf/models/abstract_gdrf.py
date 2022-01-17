import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.nn as nn
import torch

from .topic_model import SpatioTemporalTopicModel
from .utils import validate_dirichlet_param

LOGGER = logging.getLogger(__name__)


def zero_mean(x):
    return 0.0


def softmax_link_function(x):
    return torch.softmax(x, -2)


class AbstractGDRF(SpatioTemporalTopicModel):
    def __init__(
        self,
        num_observation_categories: int,
        num_topic_categories: int,
        world: List[Tuple[float, float]],
        kernel: gp.kernels.Kernel,
        dirichlet_param: Union[float, torch.Tensor],
        mean_function: Callable = None,
        link_function: Callable = None,
        device: str = "cpu",
        **kwargs
    ):
        if mean_function is None:
            mean_function = zero_mean
        if link_function is None:
            link_function = softmax_link_function
        super().__init__(
            num_observation_categories=num_observation_categories,
            num_topic_categories=num_topic_categories,
            world=world,
            device=device,
        )
        self._mean_function = mean_function
        self._kernel = kernel
        self._link_function = link_function
        if isinstance(dirichlet_param, float):
            dirichlet_param = torch.tensor(dirichlet_param)
        self._dirichlet_param = validate_dirichlet_param(
            dirichlet_param, self._K, self._V, device=self.device
        )

    def make_wt_matrix(
        self,
        dirichlet_param: Union[float, torch.Tensor],
        K: int,
        device,
        randomize: bool = False,
        randomize_metric: Optional[
            Callable[[torch.Tensor, SpatioTemporalTopicModel], float]
        ] = None,
        randomize_iters: int = 100,
    ) -> torch.Tensor:
        softmax = torch.nn.Softmax(dim=-2).float()
        ret = softmax(dirichlet_param.to(device))
        best = -1 if randomize_metric is None else randomize_metric(ret, self)
        if randomize:
            for i in range(1 if randomize_metric is None else randomize_iters):
                possible = softmax(torch.randn_like(ret))
                score = (
                    0 if randomize_metric is None else randomize_metric(possible, self)
                )
                if score > best:
                    ret = possible
        return nn.PyroParam(
            ret,
            constraint=dist.constraints.stack(
                [dist.constraints.simplex for _ in range(K)], dim=-2
            ),
        )

    @abstractmethod
    def artifacts(
        self, xs: torch.Tensor, ws: torch.Tensor, all: bool = False
    ) -> Dict[str, Any]:
        pass

    def _get(self, x: str):
        try:
            return pyro.param(self._pyro_get_fullname(x)).detach().cpu().numpy()
        except KeyError:
            return pyro.param(x).detach().cpu().numpy()

    def _getitem(self, x: str):
        return pyro.param(self._pyro_get_fullname(x)).item()

    @property
    def kernel_lengthscale(self):
        return self._get("_kernel.lengthscale")

    @property
    def kernel_variance(self):
        return self._get("_kernel.variance")

    @nn.pyro_method
    def log_topic_probs(self, xs):
        raise NotImplementedError

    @nn.pyro_method
    def topic_probs(self, xs):
        return self._link_function(self.log_topic_probs(xs)).T

    @nn.pyro_method
    def word_probs(self, xs):
        return self.topic_probs(xs) @ self.word_topic_matrix

    @nn.pyro_method
    def ml_topics(self, xs):
        return torch.argmax(self.log_topic_probs(xs), dim=-2)

    @nn.pyro_method
    def ml_words(self, xs):
        return torch.argmax(self.word_probs(xs), dim=-2)

    @nn.pyro_method
    def model(self, x, w=None, subsample=False):
        raise NotImplementedError

    @nn.pyro_method
    def guide(self, x, w=None, subsample=False):
        raise NotImplementedError

    @nn.pyro_method
    def perplexity(self, x, w):
        return ((w * self.word_probs(x).log()).sum() / -w.sum()).exp()

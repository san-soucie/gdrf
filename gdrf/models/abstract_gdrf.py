
import torch
import pyro
import pyro.nn as nn


import pyro.contrib.gp as gp

from abc import ABCMeta, abstractmethod
from .topic_model import SpatioTemporalTopicModel

from typing import Callable, Union, Any
from .utils import validate_dirichlet_param

class AbstractGDRF(SpatioTemporalTopicModel):
    def __init__(
        self,
        num_observation_categories: int,
        num_topic_categories: int,
        world: list[tuple[float, float]],
        kernel: gp.kernels.Kernel,
        dirichlet_param: Union[float, torch.Tensor],
        mean_function: Callable = lambda x: 0.0,
        link_function: Callable = lambda x: torch.softmax(x, -2),
        device: str = 'cpu'
    ):
        super().__init__(
            num_observation_categories=num_observation_categories,
            num_topic_categories=num_topic_categories,
            world=world,
            device=device
        )
        self._mean_function = mean_function
        self._kernel = kernel
        self._link_function = link_function
        self._dirichlet_param = validate_dirichlet_param(dirichlet_param, self._K, self._V, device=self.device)

    @abstractmethod
    def artifacts(self, xs: torch.Tensor, ws: torch.Tensor, all: bool = False) -> dict[str, Any]:
        pass

    def _get(self, x: str):
        return pyro.param(self._pyro_get_fullname(x)).detach().cpu().numpy()

    def _getitem(self, x: str):
        return pyro.param(self._pyro_get_fullname(x)).item()

    @property
    def kernel_lengthscale(self):
        return self._get('kernel.lengthscale')

    @property
    def kernel_variance(self):
        return self._get('kernel.variance')

    @nn.pyro_method
    def log_topic_probs(self, xs):
        raise NotImplementedError

    @nn.pyro_method
    def topic_probs(self, xs):
        return self.link_function(self.log_topic_probs(xs)).T

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
    def model(self, x, w=None):
        raise NotImplementedError

    @nn.pyro_method
    def guide(self, x, w=None):
        raise NotImplementedError

    @nn.pyro_method
    def perplexity(self, x, w):
        return ((w * self.word_probs(x).log()).sum() / -w.sum()).exp()
from abc import abstractmethod
from contextlib import contextmanager
from functools import wraps
from inspect import getfullargspec
from typing import List, Tuple

import pyro.contrib.gp
import pyro.distributions
import torch
from torch.distributions.constraint_registry import transform_to


class CategoricalModel(pyro.contrib.gp.Parameterized):
    def __init__(self, num_observation_categories: int, device: str = "cpu"):
        super().__init__()
        self._V = num_observation_categories
        self.device = device

    def _initialize_params(
        self, metric=None, extrema=min, n=100, param_init_distribution_dict=None
    ):
        if extrema not in {min, max}:
            raise ValueError(
                "extrema must be 'min' or 'max' (you provided '%s')", extrema
            )
        best_value = float("inf") if extrema is min else float("-inf")
        best_params = dict()
        for _ in range(n):
            params = self._initialize_params_once(param_init_distribution_dict)
            if metric is not None:
                value = metric()
                best_value = extrema(best_value, value)
                if best_value == value:
                    best_params = params
        if metric is not None:
            for param, val in best_params.items():
                self.__setattr__(param, val)

    def _initialize_params_once(self, param_init_distribution_dict=None):
        if param_init_distribution_dict is None:
            param_init_distribution_dict = dict()
        ret = dict()
        for param_name, (constraint, _) in self._pyro_params.items():
            if param_name in param_init_distribution_dict:
                dist = param_init_distribution_dict[param_name]
            else:
                param = self.__getattr__(param_name)
                dist = pyro.distributions.Normal(param, torch.ones_like(param))
            new_unconstrained_value = dist.sample()
            new_constrained_value = transform_to(constraint)(new_unconstrained_value)
            self.__setattr__(param_name, new_constrained_value)
            ret[param_name] = new_constrained_value
        return ret

    @property
    def V(self):
        return self._V

    @abstractmethod
    def word_probs(self, indices: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.set_mode("guide")
        return self.word_probs(indices=input)


class TopicModel(CategoricalModel):
    def __init__(
        self,
        num_observation_categories: int,
        num_topic_categories: int,
        device: str = "cpu",
    ):
        super().__init__(
            num_observation_categories=num_observation_categories, device=device
        )
        self._K = num_topic_categories

    @property
    def K(self):
        return self._K

    @abstractmethod
    def topic_probs(self, indices: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def word_topic_matrix(self) -> torch.Tensor:
        pass

    def word_probs(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.topic_probs(indices), self.word_topic_matrix)


def scale_decorator(arg_name: str):
    # https://stackoverflow.com/questions/37732639/python-decorator-access-argument-by-name
    def decorator(f):
        argspec = getfullargspec(f)
        argument_index = argspec.args.index(arg_name)

        @wraps(f)
        def wrapper(*args, **kwargs):
            self = args[0]
            try:
                value = args[argument_index]
                with self.scale_context(value) as scaled_value:
                    new_args = list(args)
                    new_args[argument_index] = scaled_value
                    return f(*new_args, **kwargs)
            except IndexError:
                value = kwargs[arg_name]
                with self.scale_context(value) as scaled_value:
                    kwargs[arg_name] = scaled_value
                    return f(*args, **kwargs)

        return wrapper

    return decorator


class SpatioTemporalTopicModel(TopicModel):
    def __init__(
        self,
        num_observation_categories: int,
        num_topic_categories: int,
        world: List[Tuple[float, float]],
        device: str = "cpu",
    ):
        super().__init__(
            num_observation_categories=num_observation_categories,
            num_topic_categories=num_topic_categories,
            device=device,
        )
        self._world = world
        self._lower_bounds = torch.tensor([b[0] for b in self._world]).to(device)
        self._upper_bounds = torch.tensor([b[1] for b in self._world]).to(device)
        self._delta_bounds = self._upper_bounds - self._lower_bounds
        self._n_dims = len(world)
        self._scaling = False

    def scale(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self._lower_bounds) / self._delta_bounds

    @contextmanager
    def scale_context(self, input: torch.Tensor):
        currently_scaling = self._scaling
        self._scaling = True
        try:
            if not currently_scaling:
                assert self._check_bounds(input)
                yield self.scale(input)
            else:
                yield input
        finally:
            self._scaling &= currently_scaling

    @property
    def dims(self):
        return self._n_dims

    def _check_dims(self, input: torch.Tensor) -> bool:
        return input.shape[-1] == self._n_dims

    def _check_bounds(self, input: torch.Tensor, epsilon: float = 1e-8) -> bool:
        return (
            self._check_dims(input)
            and (
                (input.to(self.device) - self._lower_bounds > -epsilon)
                & (input.to(self.device) - self._upper_bounds < epsilon)
            ).all()
        )

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

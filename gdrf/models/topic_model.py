import torch
import pyro
import pyro.contrib.gp

from functools import wraps
from contextlib import contextmanager
from inspect import getfullargspec

from abc import ABCMeta, abstractmethod

class CategoricalModel(pyro.contrib.gp.Parameterized):
    def __init__(self, num_observation_categories: int, device: str = 'cpu'):
        super().__init__()
        self._V = num_observation_categories
        self.device = device

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

    def __init__(self, num_observation_categories: int, num_topic_categories: int, device: str = 'cpu'):
        super().__init__(num_observation_categories=num_observation_categories, device=device)
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
        world: list[tuple[float, float]],
        device: str = 'cpu'
    ):
        super().__init__(
            num_observation_categories=num_observation_categories,
            num_topic_categories=num_topic_categories,
            device=device
        )
        self._world = world
        self._lower_bounds = torch.tensor([b[0] for b in self._world])
        self._upper_bounds = torch.tensor([b[1] for b in self._world])
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
        return self._check_dims(input) and ((input - self._lower_bounds > -epsilon) & (input - self._upper_bounds < epsilon)).all()

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass



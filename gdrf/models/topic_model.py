import torch
import pyro
import pyro.contrib.gp

from abc import ABCMeta, abstractmethod

class CategoricalModel(pyro.contrib.gp.Parameterized, metaclass=ABCMeta):
    def __init__(self, num_observation_categories: int, device: str = 'cpu'):
        self._V = num_observation_categories
        super().__init__()

    @property
    def V(self):
        return self._V

    @abstractmethod
    def word_probs(self, indices: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.set_mode("guide")
        return self.word_probs(indices=input)




class TopicModel(CategoricalModel, metaclass=ABCMeta):

    def __init__(self, num_observation_categories: int, num_topic_categories: int, device: str = 'cpu'):
        self._K = num_topic_categories
        super().__init__(num_observation_categories=num_observation_categories, device=device)

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

class SpatioTemporalTopicModel(TopicModel, metaclass=ABCMeta):

    def __init__(
        self,
        num_observation_categories: int,
        num_topic_categories: int,
        world: list[tuple[float]],
        device: str = 'cpu'
    ):
        self._world = world
        self._lower_bounds = torch.tensor([b[0] for b in self._world])
        self._upper_bounds = torch.tensor([b[1] for b in self._world])
        self._n_dims = len(world)
        super().__init__(
            num_observation_categories=num_observation_categories,
            num_topic_categories=num_topic_categories,
            device=device
        )

    def _check_dims(self, input: torch.Tensor) -> bool:
        return input.shape[-1] == self._n_dims

    def _check_bounds(self, input: torch.Tensor) -> bool:
        return self._check_dims(input) and ((input - self._lower_bounds > 0) & (input - self._upper_bounds < 0)).all()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_bounds(input)
        return super().forward(input)



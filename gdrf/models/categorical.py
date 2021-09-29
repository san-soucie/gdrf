from .topic_model import CategoricalModel


class SimpleCategoricalModel(CategoricalModel):
    def __init__(self, num_observation_categories: int):
        super().__init__(num_observation_categories)

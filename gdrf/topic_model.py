import pyro
import pyro.distributions as dist
import pyro.nn as nn
import torch
import pyro.contrib.gp as gp


class TopicModel(gp.Parameterized):
    def __init__(self,
                 document_topic_module: torch.nn.Module,
                 n_vocab: int,
                 n_topic: int
                 ):
        super().__init__()
        self._word_topic_matrix = torch.nn.Parameter(torch.ones(n_topic, n_vocab) / n_vocab)
        self._word_topic_matrix = nn.PyroParam(self._word_topic_matrix, constraint=dist.constraints.stack([dist.constraints.simplex for _ in range(n_topic)], dim=-2))
        self._document_topic_module = document_topic_module

    def forward(self, document_indices: torch.Tensor):
        document_topic_matrix = self._document_topic_module(document_indices)
        document_word_matrix = document_topic_matrix @ self._word_topic_matrix
        return document_word_matrix


class DocumentTopicMatrix(gp.Parameterized):
    def __init__(self,
                 document_size: torch.Size,
                 n_topic: int
                 ):
        super().__init__()
        self._topic_probs = torch.nn.Parameter(torch.ones(*document_size, n_topic) / n_topic)
        self.n_documents = document_size.numel()
        self._topic_probs = nn.PyroParam(self._topic_probs, constraint=dist.constraints.stack([dist.constraints.simplex for _ in range(document_size[-1])], dim=-2))

    def topic_probs(self, document_indices: torch.Tensor):
        return self._topic_probs.gather(-1, document_indices[..., None])[..., 0]

    def forward(self, document_indices):
        return self.topic_probs(document_indices)

dsize = torch.Size([25, 80])
n_topic = 5
n_vocab = 49
dt_mat = DocumentTopicMatrix(document_size=dsize, n_topic=n_topic)
tmodel = TopicModel(dt_mat, n_vocab, n_topic)
print(tmodel(torch.tensor([[1, 3, 5], [2, 6, 1]]).T).shape)

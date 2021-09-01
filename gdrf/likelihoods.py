import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.contrib.gp.likelihoods.likelihood import Likelihood
import torch
from torch import Tensor, Size
from torch.distributions import constraints
from pyro.nn.module import PyroParam


def _softmax(x):
    return F.softmax(x, dim=-1)


class MultiClass_Dirichlet(Likelihood):
    """
    Implementation of MultiClass likelihood, which is used for multi-class
    classification problems.

    MultiClass likelihood uses :class:`~pyro.distributions.Categorical`
    distribution, so ``response_function`` should normalize its input's rightmost axis.
    By default, we use `softmax` function.

    :param int num_classes: Number of classes for prediction.
    :param callable response_function: A mapping to correct domain for MultiClass
        likelihood.
    :param Union[float, torch.Tensor] dirichlet_param: Dirichlet parameter
    """

    def __init__(self, num_classes, num_latent_topics, response_function=None, dirichlet_param=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_latent_topics = num_latent_topics
        self.response_function = (
            _softmax if response_function is None else response_function
        )
        self.dirichlet_param = (
            Tensor([dirichlet_param] * num_classes) if type(dirichlet_param) is float else dirichlet_param
        )
    @pyro.infer.config_enumerate
    def forward(self, f_loc, f_var, w=None):
        r"""
        Samples :math:`w` given :math:`f_{loc}`, :math:`f_{var}` according to

            .. math:: f & \sim \mathbb{Normal}(f_{loc}, f_{var}),\\
                p(w|z) & \sim \mathbb{Dir}(\dirichlet_param) \\
                z & \sim \mathbb{Categorical}(f)
                w & \sim p(w|z).

        .. note:: The log likelihood is estimated using Monte Carlo with 1 sample of
            :math:`f`.

        :param torch.Tensor f_loc: Mean of latent function output.
        :param torch.Tensor f_var: Variance of latent function output.
        :param torch.Tensor y: Training output tensor.
        :returns: a tensor sampled from likelihood
        :rtype: torch.Tensor
        """
        # calculates Monte Carlo estimate for E_q(f) [logp(y | f)]
        f = dist.Normal(f_loc, f_var.sqrt())()
        if f.dim() < 2:
            raise ValueError(
                "Latent function output should have at least 2 "
                "dimensions: one for number of classes and one for "
                "number of data."
            )

        # swap class dimension and data dimension
        f_swap = f.transpose(-2, -1)  # -> num_data x num_classes
        if f_swap.size(-1) != self.num_latent_topics:
            raise ValueError(
                "Number of Gaussian processes should be equal to the "
                "number of latent topics. Expected {} but got {}.".format(
                    self.num_latent_topics, f_swap.size(-1)
                )
            )
        if self.response_function is _softmax:
            z_dist = dist.Categorical(logits=f_swap)
        else:
            f_res = self.response_function(f_swap)
            z_dist = dist.Categorical(f_res)
        p_w_z_dist = dist.Dirichlet(self.dirichlet_param).expand(Size([self.num_latent_topics])).to_event(1)
        phi = pyro.sample(self._pyro_get_fullname("phi"), p_w_z_dist)  # K x V
        with pyro.plate("obs"):
            z = pyro.sample(self._pyro_get_fullname("z"), z_dist)
            w_dist = dist.Categorical(pyro.ops.indexing.Vindex(phi)[z, :])
            # print(w_dist.shape())
            # if w is not None:
            #     w_dist = w_dist.expand_by(w.shape[: -f.dim() + 1]).to_event(w.dim())
            return pyro.sample(self._pyro_get_fullname("w"), w_dist, obs=w, infer={'is_auxiliary': True})


class MultinomialDirichlet(Likelihood):
    """
    Implementation of MultiClass likelihood, which is used for multi-class
    classification problems.

    MultiClass likelihood uses :class:`~pyro.distributions.Categorical`
    distribution, so ``response_function`` should normalize its input's rightmost axis.
    By default, we use `softmax` function.

    :param int num_classes: Number of classes for prediction.
    :param int num_input_dims: Number of Gaussian process input dimensions
    :param callable response_function: A mapping to correct domain for MultiClass
        likelihood.
    """
    def __init__(self, num_classes, num_input_dims, response_function=None):
        super(MultinomialDirichlet, self).__init__()
        self.num_input_dims = num_input_dims
        self.num_classes = num_classes
        self.dirichlet_matrix = PyroParam(torch.ones(num_input_dims, num_classes) / num_classes, constraints.stack([constraints.simplex for _ in range(num_input_dims)]))
        self.response_function = _softmax if response_function is None else response_function

    def forward(self, f_loc, f_var, y=None):
        r"""
        Samples :math:`y` given :math:`f_{loc}`, :math:`f_{var}` according to

            .. math:: f & \sim \mathbb{Normal}(f_{loc}, f_{var}),\\
                y & \sim \mathbb{Categorical}(f).

        .. note:: The log likelihood is estimated using Monte Carlo with 1 sample of
            :math:`f`.

        :param torch.Tensor f_loc: Mean of latent function output.
        :param torch.Tensor f_var: Variance of latent function output.
        :param torch.Tensor y: Training output tensor.
        :returns: a tensor sampled from likelihood
        :rtype: torch.Tensor
        """
        # calculates Monte Carlo estimate for E_q(f) [logp(y | f)]
        f = dist.Normal(f_loc, f_var.sqrt())()
        if f.dim() < 2:
            raise ValueError("Latent function output should have at least 2 "
                             "dimensions: one for number of classes and one for "
                             "number of data.")

        # swap class dimension and data dimension
        f_swap = f.transpose(-2, -1)  # -> num_data x num_classes
        if f_swap.size(-1) != self.num_classes:
            raise ValueError("Number of Gaussian processes should be equal to the "
                             "number of classes. Expected {} but got {}."
                             .format(self.num_classes, f_swap.size(-1)))
        f_res = self.response_function(f_swap) @ self.dirichlet_matrix
        y_dist = dist.Multinomial(f_res, validate_args=False)
        if y is not None:
            y_dist = y_dist.expand_by(y.shape[:-f_res.dim() + 1]).to_event(y.dim())
        return pyro.sample("y", y_dist, obs=y)

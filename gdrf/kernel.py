import torch
import pyro.nn as nn
from torch.distributions import constraints

class Kernel(nn.module.PyroModule):
    """
    Base class for kernels used in this Gaussian Process module.

    Every inherited class should implement a :meth:`forward` pass which takes inputs
    :math:`X`, :math:`Z` and returns their covariance matrix.

    To construct a new kernel from the old ones, we can use methods :meth:`add`,
    :meth:`mul`, :meth:`exp`, :meth:`warp`, :meth:`vertical_scale`.

    References:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param int input_dim: Number of feature dimensions of inputs.
    :param torch.Tensor variance: Variance parameter of this kernel.
    :param list active_dims: List of feature dimensions of the input which the kernel
        acts on.
    """

    def __init__(self, input_dim, active_dims=None):
        super().__init__()
        if active_dims is None:
            active_dims = list(range(input_dim))
        elif input_dim != len(active_dims):
            raise ValueError(
                "Input size and the length of active dimensionals should be equal."
            )
        self.input_dim = input_dim
        self.active_dims = active_dims

    def forward(self, X, Z=None, diag=False):
        r"""
        Calculates covariance matrix of inputs on active dimensionals.

        :param torch.Tensor X: A 2D tensor with shape :math:`N \times input\_dim`.
        :param torch.Tensor Z: An (optional) 2D tensor with shape
            :math:`M \times input\_dim`.
        :param bool diag: A flag to decide if we want to return full covariance matrix
            or just its diagonal part.
        :returns: covariance matrix of :math:`X` and :math:`Z` with shape
            :math:`N \times M`
        :rtype: torch.Tensor
        """
        raise NotImplementedError


    def _slice_input(self, X):
        r"""
        Slices :math:`X` according to ``self.active_dims``. If ``X`` is 1D then returns
        a 2D tensor with shape :math:`N \times 1`.

        :param torch.Tensor X: A 1D or 2D input tensor.
        :returns: a 2D slice of :math:`X`
        :rtype: torch.Tensor
        """
        if X.dim() == 2:
            return X[:, self.active_dims]
        elif X.dim() == 1:
            return X.unsqueeze(1)
        else:
            raise ValueError("Input X must be either 1 or 2 dimensional.")

def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


class Isotropy(Kernel):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    By default, the parameter ``lengthscale`` has size 1. To use the isotropic version
    (different lengthscale for each dimension), make sure that ``lengthscale`` has size
    equal to ``input_dim``.

    :param torch.Tensor lengthscale: Length-scale parameter of this kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = variance

        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = lengthscale

    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        scaled_X = X / self.lengthscale
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X ** 2).sum(1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)

    def _scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))

    def _diag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active features.
        """
        return self.variance.expand(X.size(0))

class RBF(Isotropy):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \sigma^2\exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r2 = self._square_scaled_dist(X, Z)
        return self.variance * torch.exp(-0.5 * r2)




class Exponential(Isotropy):
    r"""
    Implementation of Exponential kernel:

        :math:`k(x, z) = \sigma^2\exp\left(-\frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r = self._scaled_dist(X, Z)
        return self.variance * torch.exp(-r)



class Matern32(Isotropy):
    r"""
    Implementation of Matern32 kernel:

        :math:`k(x, z) = \sigma^2\left(1 + \sqrt{3} \times \frac{|x-z|}{l}\right)
        \exp\left(-\sqrt{3} \times \frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r = self._scaled_dist(X, Z)
        sqrt3_r = 3 ** 0.5 * r
        return self.variance * (1 + sqrt3_r) * torch.exp(-sqrt3_r)



class Matern52(Isotropy):
    r"""
    Implementation of Matern52 kernel:

        :math:`k(x,z)=\sigma^2\left(1+\sqrt{5}\times\frac{|x-z|}{l}+\frac{5}{3}\times
        \frac{|x-z|^2}{l^2}\right)\exp\left(-\sqrt{5} \times \frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r2 = self._square_scaled_dist(X, Z)
        r = _torch_sqrt(r2)
        sqrt5_r = 5 ** 0.5 * r
        return self.variance * (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r)

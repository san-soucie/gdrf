from abc import ABC, abstractmethod
import torch

from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import get_dependent_plate_dims, is_validation_enabled, torch_sum, torch_item
from pyro.util import check_if_enumerated, warn_if_nan
from typing import Callable, Union
import warnings
from .divergence import Divergence
from .loss import Loss


class GeneralizedVariationalLoss(ELBO):
    def __init__(self,
                 divergence: Divergence,
                 loss_fn: Loss,
                 num_particles=2,
                 max_plate_nesting=float("inf"),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=False,
                 strict_enumeration_warning=True,
                 ):
        self.loss_fn = loss_fn
        self.divergence = divergence
        if max_iarange_nesting is not None:
            warnings.warn(
                "max_iarange_nesting is deprecated; use max_plate_nesting instead",
                DeprecationWarning,
            )
            max_plate_nesting = max_iarange_nesting
        super().__init__(
            num_particles=num_particles,
            max_plate_nesting=max_plate_nesting,
            vectorize_particles=vectorize_particles,
            strict_enumeration_warning=strict_enumeration_warning,
        )

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return self.loss_fn.loss(model, guide, *args, **kwargs) + self.divergence.loss(model, guide, *args, **kwargs)

    def loss_and_grads(self, model, guide, *args, **kwargs):
        return self.loss_fn.loss_and_grads(model, guide, *args, **kwargs) + self.divergence.loss_and_grads(model, guide, *args, **kwargs)


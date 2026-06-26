import numpy as np
import BayesBoom.boom as boom
import BayesBoom.R as R
from ..boom_utils import to_boom_vector, to_boom_spd


class MultinomialLogitSpikeSlabPrior:
    def __init__(self):
        pass

    @classmethod
    def from_model(cls,
                   model: boom.MultinomialLogitModel,
                   expected_model_size=1.0,
                   diagonal_shrinkage=.05):
        obj = cls()
        obj._prior_precision = model.xtx.to_numpy() / model.sample_size
        obj._prior_precision = (
            (1 - diagonal_shrinkage) * obj._prior_precision
            + diagonal_shrinkage * np.diag(np.diag(obj._prior_precision))
        )
        xdim = obj._prior_precision.shape[1]
        obj._prior_inclusion_probabilities = np.full(
            xdim, expected_model_size / xdim)
        return obj

    @property
    def slab(self):
        xdim = self._prior_precision.shape[1]
        return boom.MvnModel(
            R.to_boom_vector(np.zeros(xdim)),
            R.to_boom_spd(self._prior_precision),
            ivar=True)

    @property
    def spike(self):
        return boom.VariableSelectionPrior(R.to_boom_vector(
            self._prior_inclusion_probabilities))

    def create_sampler(self, model, assign=True):
        sampler = boom.MultinomialLogitCompositeSpikeSlabSampler(
            model,
            self.slab,
            self.spike)
        if assign:
            model.set_method(sampler)
        return sampler

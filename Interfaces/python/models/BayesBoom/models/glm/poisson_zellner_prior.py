import numpy as np
import BayesBoom.boom as boom
from .binomial_logit_model import LogitZellnerPrior


def trimmed_log(x):
    if x > 1e-5:
        return np.log(x)
    else:
        return np.float64(-11.512925464970229)
    

class PoissonZellnerPrior:
    def __init__(self,
                 predictors,
                 counts=None,
                 exposure=None,
                 prior_event_rate=1.0,
                 expected_model_size=1.0,
                 prior_information_weight=1.0,
                 diagonal_shrinkage=.5,
                 optional_coefficient_estimate=None,
                 max_flips=-1,
                 prior_inclusion_probabilities=None):
        self._max_flips = max_flips
        sample_size = predictors.shape[0]
        xdim = predictors.shape[1]

        xtx = predictors.T @ predictors * prior_information_weight / sample_size
        xtx_diagonal = np.diagonal(xtx).copy()
        xtx *= 1 - diagonal_shrinkage
        np.fill_diagonal(xtx, xtx_diagonal)

        self._precision = xtx
        self._mean = np.zeros(xdim)
        if counts is None:
            self._mean[0] = np.log(prior_event_rate)
        else:
            if exposure is None:
                exposure = np.ones(sample_size)
            p_hat = np.nanmean(counts / exposure)
            self._mean[0] = trimmed_log(p_hat)
        if not np.isfinite(self._mean[0]):
            self._mean[0] = 0.0

        if prior_inclusion_probabilities is None:
            prior_inclusion_probabilities = np.full(
                xdim, expected_model_size / xdim)
        self._prior_inclusion_probabilities = prior_inclusion_probabilities

    @classmethod
    def from_parameters(self, mean, precision, prior_inclusion_probabilities,
                        max_flips=-1):
        xdim = len(mean)
        predictors = np.random.randn(xdim, xdim)
        y = np.full(xdim, 0.5)
        trials = np.ones(xdim)
        ans = LogitZellnerPrior(predictors, y, trials, max_flips=max_flips)
        ans._prior_inclusion_probabilities = prior_inclusion_probabilities
        ans._mean = mean
        ans._precision = precision
        return ans

    @property
    def slab(self):
        return boom.MvnModel(
            boom.Vector(self._mean),
            boom.SpdMatrix(self._precision),
            True)

    @property
    def spike(self):
        return boom.VariableSelectionPrior(
            self._prior_inclusion_probabilities)

    @property
    def max_flips(self):
        return self._max_flips

    def create_sampler(self, model, assign=False):
        if not isinstance(model, boom.PoissonRegressionModel):
            raise Exception(
                "Expected 'model' to be a boom.PoissonRegressionModel.")
        sampler = boom.PoissonRegressionSpikeSlabSampler(
            model=model,
            slab=self.slab,
            spike=self.spike,
            clt_threshold=5,
            seeding_rng=boom.GlobalRng.rng)
        if self._max_flips > 0 and np.isfinite(self._max_flips):
            sampler.limit_model_selection(self._max_flips)
        if assign:
            model.set_method(sampler)
        return sampler



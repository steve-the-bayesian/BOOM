import numpy as np
import BayesBoom.boom as boom
import BayesBoom.R as R
from ..UniformModel import UniformModel
from ..boom_utils import (
    to_boom_vector,
    to_boom_matrix,
    to_boom_spd,
)
from .regression_spike_slab_prior import RegressionSpikeSlabPrior

class StudentSpikeSlabPrior(RegressionSpikeSlabPrior):
    """
    A SpikeSlabPrior appropriate for regression models with Student T errors.
    """
    def __init__(
            self,
            x,
            y=None,
            expected_r2=.5,
            prior_df=.01,
            expected_model_size=1,
            prior_information_weight=.01,
            diagonal_shrinkage=.5,
            optional_coefficient_estimate=None,
            max_flips=-1,
            mean_y=None,
            sdy=None,
            prior_inclusion_probabilities=None,
            sigma_upper_limit=np.inf,
            tail_thickness_prior=UniformModel(0.1, 100)
    ):
        """
        Args:
          tail_thickness_prior: An object with a boom() method, which returns a
            boom.DoubleModel describing the prior on the tail thickness
            parameter.

          All other arguments are as documented in RegressionSpikeSlabPrior.
        """
        super().__init__(
            x=x,
            y=y,
            expected_r2=expected_r2,
            prior_df=prior_df,
            expected_model_size=expected_model_size,
            prior_information_weight=prior_information_weight,
            diagonal_shrinkage=diagonal_shrinkage,
            optional_coefficient_estimate=optional_coefficient_estimate,
            max_flips=max_flips,
            mean_y=mean_y,
            sdy=sdy,
            prior_inclusion_probabilities=prior_inclusion_probabilities,
            sigma_upper_limit=sigma_upper_limit)
        self._nu_prior = tail_thickness_prior

    @property
    def tail_thickness(self):
        """
        A boom.DoubleModel giving the prior distribution on the tail thickness
        parameter.
        """
        return self._nu_prior.boom()

    def create_sampler(self, model, assign=False):
        pass


import BayesBoom.boom as boom
from .boom_py_utils import to_boom_matrix, to_boom_vector
import numpy as np
import pandas as pd


def assign_classes(posterior_class_probabilities,
                   global_target=None,
                   max_kl=1.0):
    """
    Assign class memberships to a collection of object, while maintaining an
    empirical distribution of assigned classes close to a global target
    distribution.

    Args:
      posterior_class_probabilities: A Matrix, numpy array, or DataFrame, with
        each row corresponding to an object, and each column to a potential
        class value.  posterior_class_probabilities[i, j] is the posterior
        probability that object i belongs to class j.
      global_target: A vector containing the discrete probability
        distribution of class membership for the population.  If None, then the
        global target is taken as the (row-)mean of the
        posterior_class_probabilities matrix.
      max_kl: The largest tolerable Kullback-Liebler divergence between the
        empirical distribution of the assigned values, and the global target
        distribution.

    Returns:
      A list of integers indicating the class assignment for each object.
    """
    if global_target is None:
        global_target = np.mean(posterior_class_probabilities, axis=0)

    assigner = ClassAssigner()
    assigner.set_max_kl(max_kl)
    return assigner.assign(posterior_class_probabilities, global_target)


class ClassAssigner:
    """
    Assign class memberships to a collection of object, while maintaining an
    empirical distribution of assigned classes close to a global target
    distribution.

    A wrapper for the boom.ClassAssigner.  Wrapping allows care to be taken with
    input types.
    """

    def __init__(self):
        self._assigner = boom.ClassAssigner()

    def set_initial_temperature(self, temp):
        """
        Set the temperature to use at the start of a simulated annealing run.
        """
        self._assigner.set_initial_temperature(float(temp))

    def set_max_kl(self, kl):
        """
        Set the largest tolerable Kullback-Liebler divergence between the
        empirical distribution of the assigned values, and the global
        target distribution.
        """
        self._assigner.set_max_kl(float(kl))

    def set_max_iterations(self, niter):
        """
        Set the maximum number of simulated annealing iterations (function
        evaluations) per SA run.
        """
        self._assigner.set_max_iterations(int(niter))

    def assign(self,
               marginal_posteriors,
               global_target,
               rng=boom.GlobalRng.rng):
        """
        Assign class memberships to a collection of object, while maintaining an
        empirical distribution of assigned classes close to a global target
        distribution.

        Args:
          marginal_posteriors: A matrix, numpy array, or DataFrame with each row
            corresponding to an object, and each column to a potential class
            value. marginal_posteriors[i, j] is the posterior probability that
            object i belongs to class j.
          global_target: A boom.Vector containing the discrete probability
            distribution of class membership for the  population.
          rng:  A boom.RNG used to drive the simulated annealing algorithm.

        Returns:
          A pd.Series of integers indicating the class assignment for each
          object.
        """
        return_index = getattr(marginal_posteriors, "index", None)

        ans = self._assigner.assign(
            to_boom_matrix(marginal_posteriors),
            to_boom_vector(global_target),
            rng=rng)

        return pd.Series(ans, index=return_index, dtype="int")

    @property
    def kl(self):
        """
        The Kullback-Liebler divergence between the most recently produced
        set of assignments and the global target distribution.
        """
        return self._assigner.kl

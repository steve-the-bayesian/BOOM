import BayesBoom as Boom
import patsy
import spikeslab as ss


class Bsts:
    """A Bayesian structural time series model.

    """

    def __init__(self, family="gaussian", prior=None, seed=None):
        self._family = family
        assert family in set(["gaussian", "poisson", "binomial", "student"])
        self._model = None
        self._state_models = []

    def add_local_level(self):
        pass

    def add_local_linear_trend(self):
        pass

    def add_seasonal(self):
        pass

    def train(self, formula, data, niter, ping):
        self._format_data(formula, data)

        for i in range(niter):
            self._model.sample_posterior()
            self._record_state(i)

    def _record_state(self, i):
        """Record the state from the
        """
        for m in self._state_models:
            m.record_state(i)
        self._model

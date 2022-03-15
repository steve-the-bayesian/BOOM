import BayesBoom.boom as boom
import numpy as np
import BayesBoom.R as R
from .state_models import StateModel
import patsy


class DynamicRegressionStateModel(StateModel):
    """
    The state in this model represents a vector of regression coefficients that
    evolves over time.  The state equation is thus

       y[t] = beta[t].dot(x[t]) + other_state[t] + error[t]

    The variables x[t] are determined by a model formula and a data set
    supplied at contruction time.

    The state transition equation is a random walk.

       beta[t+1] = beta[t] + error

    The vector of errors is independent zero-mean Gaussian noise, where
    component i has standard deviation sigma[i].
    """

    def __init__(self,
                 formula,
                 data,
                 sigma_prior=None,
                 initial_state_prior=None):
        """
        Args:

          formula: A string describing a regression relationship in the R
            modeling language.  For example "y ~ x1 + x3" or "y ~ x1 * x2 + x3".
            The 'dot' operator is not understood, but "y ~ " + dot(
        """
        self._formula = formula + "- 1"
        response, self._predictors = patsy.dmatrices(self._formula, data=data)
        xtx = self._predictors.T @ self._predictors
        xty = self._predictors.T @ response
        sdy = np.nanstd(response)
        sdx = np.nanstd(self._predictors, axis=0)
        self._sigma_prior = self._verify_prior(sigma_prior, sdy, sdx)
        self._initial_state_prior = self._verify_initial_state_prior(
            initial_state_prior, xtx, xty, sdy)
        self._build_state_model()

    def __repr__(self):
        ans = f"DynamicRegressionStateModel: {self._formula}"
        return ans

    @property
    def label(self):
        if len(self._formula) > 40:
            return "DynReg(long formula)"
        else:
            return f"DynReg({self._formula})"

    @property
    def state_dimension(self):
        return self._predictors.shape[1]

    @property
    def xdim(self):
        return self.state_dimension

    def allocate_space(self, niter, time_dimension):
        nrows = self._predictors.shape[0]
        if time_dimension != nrows:
            raise Exception(f"time_dimension = {time_dimension}, but "
                            f"self._predictors has {nrows} rows.")
        self.sigma_draws = np.zeros((niter, self.xdim))
        self._state_contribution = np.zeros((niter, time_dimension))
        self._dynamic_coefficients = np.zeros((
            niter, self.xdim, time_dimension))

    def record_state(self, iteration, state_matrix):
        if self._state_index < 0:
            raise Exception("Each state model must be told where its state"
                            "component begins in the global state vector.  "
                            "Try calling set_state_index.")
        self.sigma_draws[iteration, :] = self._state_model.sigma.to_numpy()

        begin = self._state_index
        end = self._state_index + self.state_dimension
        self._dynamic_coefficients[iteration, :, :] = state_matrix[begin:end, :]
        self._state_contribution[iteration, :] = (
            self._dynamic_coefficients[iteration, :, :] * self._predictors.T
            ).sum(axis=0)

    def restore_state(self, iteration):
        self._state_model.set_sigma(self.sigma_draws[iteration, :])

    @property
    def state_contribution(self):
        return self._state_contribution

    def __getstate__(self):
        payload = self.__dict__.copy()
        del payload["_state_model"]
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        self._build_state_model()

    def _validate_initial_distributions(self, initial_state_prior, y, sdy):
        if initial_state_prior is None:
            if sdy is None:
                sdy = np.nanstd(y, ddof=1)
            initial_state_prior = R.NormalPrior(0.0, float(sdy))
        if not isinstance(initial_state_prior, R.NormalPrior):
            raise Exception(
                "initial_state_prior should be an R.NormalPrior.")
        self._initial_state_prior = initial_state_prior

    def _build_state_model(self):
        self._state_model = boom.DynamicRegressionStateModel(
            R.to_boom_matrix(self._predictors))

        boom_sigma_priors = [pri.boom() for pri in self._sigma_prior]
        state_model_sampler = boom.DynamicRegressionIndependentPosteriorSampler(
            self._state_model, boom_sigma_priors)
        for i, prior in enumerate(self._sigma_prior):
            finite_limit = np.isfinite(prior.upper_limit)
            if prior.upper_limit > 0 and finite_limit:
                state_model_sampler.set_sigma_max(i, prior.upper_limit)
        self._state_model.set_method(state_model_sampler)

        self._state_model.set_initial_state_mean(
            R.to_boom_vector(self._initial_state_prior.mean))
        self._state_model.set_initial_state_variance(
            R.to_boom_spd(self._initial_state_prior.Sigma))

    def _verify_prior(self, sigma_prior, sdy, sdx):
        if sigma_prior is None:
            self._sigma_prior = [
                R.SdPrior(.01 * sdy / sdxi, 1) for sdxi in sdx]
        elif isinstance(sigma_prior, R.SdPrior):
            self._sigma_prior = [sigma_prior] * len(sdx)

        if not R.is_iterable(self._sigma_prior) and all(
            [isinstance(x, R.SdPrior) for x in self._sigma_prior]
        ):
            raise Exception(
                "sigma_prior must be a list-like of R.SdPrior objects."
            )

        return self._sigma_prior

    def _verify_initial_state_prior(
            self, initial_state_prior, xtx, xty, sdy):
        if initial_state_prior is None:
            try:
                beta_hat = np.linalg.solve(xtx, xty)
                if not np.all(np.finite(beta_hat)):
                    raise Exception("Least squares initializer failed.")

                self._initial_state_prior = R.MvnPrior(
                    beta_hat, sdy * sdy * np.linalg.inv(xtx))
            except Exception:
                self._initial_state_prior = R.MvnPrior(
                    np.zeros(self.xdim),
                    sdy * sdy * np.diag(1.0 / np.diagonal(xtx)))

        elif isinstance(initial_state_prior, R.NormalPrior):
            mean = np.full(initial_state_prior.mean, self.xdim)
            var = np.full(initial_state_prior.sd ** 2, self.xdim)
            self._initial_state_prior = R.MvnPrior(mean, np.diag(var))

        elif isinstance(initial_state_prior, list) and all(
                [isinstance(x, R.NormalPrior) for x in initial_state_prior]):
            mean = np.array([x.mean for x in initial_state_prior])
            var = np.array([x.sd ** 2 for x in initial_state_prior])
            self._initial_state_prior = R.MvnPrior(mean, np.diag(var))

        else:
            if not isinstance(initial_state_prior, R.MvnPrior):
                raise Exception("Unrecognized type for initial_state_prior.")
            self._initial_state_prior = initial_state_prior

        return self._initial_state_prior

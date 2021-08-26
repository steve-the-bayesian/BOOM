import unittest
import numpy as np
import pandas as pd
import BayesBoom.boom as boom
import BayesBoom.dynreg as dynreg
import BayesBoom.spikeslab as ss
import BayesBoom.R as R
import matplotlib.pyplot as plt

import pdb


class TestDynamicRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    @staticmethod
    def simulate_data_from_model(time_dimension: int,
                                 typical_sample_size: int,
                                 xdim: int,
                                 residual_sd: float,
                                 unscaled_innovation_sd: np.ndarray,
                                 p00: np.ndarray,
                                 p11: np.ndarray):
        from BayesBoom.R import rmarkov
        inclusion = np.full((xdim, time_dimension), -1)
        p00 = p00.ravel()
        p11 = p11.ravel()
        for j in range(xdim):
            P = np.array(
                [
                    [p00[j], 1 - p00[j]],
                    [1 - p11[j], p11[j]]
                ]
            )
            inclusion[j, :] = rmarkov(time_dimension, P)

        coefficients = np.zeros((xdim, time_dimension))
        for j in range(xdim):
            sd = unscaled_innovation_sd[j] * residual_sd
            for t in range(time_dimension):
                prev = 0 if t == 0 else coefficients[j, t-1]
                coefficients[j, t] = inclusion[j, t] * (
                    prev + np.random.randn(1) * sd
                )

        data = []
        xnames = ["X" + str(i) for i in range(xdim)]
        xnames[0] = "(Intercept)"
        for t in range(time_dimension):
            sample_size = np.random.poisson(typical_sample_size, 1)[0]
            X = np.random.randn(sample_size, xdim)
            X[:, 0] = 1.0
            yhat = X @ coefficients[:, t]
            y = yhat + residual_sd * np.random.randn(sample_size)
            chunk = pd.DataFrame(X, columns=xnames)
            chunk["y"] = y
            chunk["timestamp"] = t
            data.append(chunk)

        full_data = pd.concat(data, axis=0, ignore_index=True)
        return full_data, coefficients, inclusion

    def test_mcmc(self):
        xdim = 4
        true_residual_sd = .25
        data, coefficients, inclusion = self.simulate_data_from_model(
            time_dimension=200,
            typical_sample_size=5000,
            xdim=xdim,
            residual_sd=true_residual_sd,
            unscaled_innovation_sd=np.array([.01] * xdim),
            p00=np.array([.95] * xdim),
            p11=np.array([.99] * xdim),
        )
        model = dynreg.SparseDynamicRegressionModel(
            "y ~ " + ss.dot(data, ["y", "timestamp", "(Intercept)"]),
            data=data,
            timestamps="timestamp",
            niter=100,
            residual_precision_prior=R.SdPrior(true_residual_sd, 1),
            seed=8675309)

        model.plot()
        for i in range(4):
            self.assertEqual("", boom.check_stochastic_process(
                boom.Matrix(model._beta_draws[:, i, :]),
                boom.Vector(coefficients[i, :]),
                confidence=.95,
                sd_ratio_threshold=10000,  # Turn off the sd_ratio check.
            ))

        posterior_mean_residual_sd = np.mean(model._residual_sd_draws[10:])
        self.assertGreater(posterior_mean_residual_sd, true_residual_sd - .02)
        self.assertLess(posterior_mean_residual_sd, true_residual_sd + .02)

        sd_fig, sd_ax = plt.subplots(1, 2)
        model.plot_residual_sd(ax=sd_ax[0])
        model.plot_residual_sd(ax=sd_ax[1], type="ts")
        # sd_fig.show()

        size_fig, size_ax = plt.subplots(1, 1)
        model.plot_size(ax=size_ax)
        # size_fig.show()

        # fig, ax = plt.subplots(2, 2)
        # which = 0
        # for i in range(2):
        #     for j in range(2):
        #         R.plot_dynamic_distribution(model._beta_draws[:, which, :],
        #                                     ax=ax[i, j])
        #         R.lines(np.arange(coefficients.shape[1]),
        #                 coefficients[which, :], color="blue", ax=ax[i, j])
        #         which += 1
        # fig.show()


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
#    import warnings
#    warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestDynamicRegression()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_mcmc()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

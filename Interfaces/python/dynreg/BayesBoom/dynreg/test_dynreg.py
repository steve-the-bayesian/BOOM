import unittest
import numpy as np
import pandas as pd
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
        data, coefficients, inclusion = self.simulate_data_from_model(
            time_dimension=200,
            typical_sample_size=50,
            xdim=xdim,
            residual_sd=.25,
            unscaled_innovation_sd=np.array([.01] * xdim),
            p00=np.array([.95] * xdim),
            p11=np.array([.99] * xdim),
        )
        model = dynreg.SparseDynamicRegressionModel(
            "y ~ " + ss.dot(data, ["y", "timestamp", "(Intercept)"]),
            data=data,
            timestamps="timestamp",
            niter=100)
        import pdb
        pdb.set_trace()
        print("foo")


_debug_mode = True

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

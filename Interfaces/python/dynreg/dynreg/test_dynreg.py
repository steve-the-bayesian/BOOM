import unittest
import numpy as np
import BayesBoom.boom as boom


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
        for t in range(time_dimension):
            sample_size = np.random.poisson(typical_sample_size, 1)[0]
            X = np.random.randn(sample_size, xdim)
            X[:, 0] = 1.0
            yhat = X @ coefficients[:, t]
            y = yhat + residual_sd * np.random.randn(sample_size)
            data.append(boom.RegressionDataTimePoint(boom.Matrix(X),
                                                     boom.Vector(y)))
        return data, coefficients, inclusion

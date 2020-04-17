import unittest
import BayesBoom as boom
import numpy as np
import R

class DynregTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self._residual_sd = 0.2
        self._unscaled_innovation_sd = np.array([.1, .15, .2])
        self._xdim = 3
        self._p00 = np.array([.8, .7, .6])
        self._p11 = np.array([.7, .8, .9])

    def simulate_null_data(self, time_dimension, typical_sample_size, xdim):
        ans = []
        sample_sizes = np.random.poisson(typical_sample_size, time_dimension)
        for i in range(time_dimension):
            time_point = boom.RegressionDataTimePoint(xdim)
            sample_size = sample_sizes[i]
            responses = np.random.randn(sample_size)
            predictors = np.random.randn(sample_size, xdim)
            for j in range(sample_size):
                reg_data = boom.RegressionData(responses[i], predictors[i, :])
                time_point.add_data(reg_data)
            ans.append(time_point)
        return ans

    def simulate_data_from_model(self,
                                 time_dimension: int,
                                 typical_sample_size: int,
                                 xdim: int,
                                 residual_sd: float,
                                 unscaled_innovation_sd: np.ndarray,
                                 p00: np.ndarray,
                                 p11: np.ndarray):
        from R import rmarkov
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
            for t in range(time_dimension):
                prev = 0 if t == 0 else coefficients[j, t-1]
                coefficients[j, t] = (
                    prev +
                    np.random.randn(1) * residual_sd * unscaled_innovation_sd[j]
                )
                coefficients[j, t] *= inclusion[j, t]

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

    def setup_model(self, data, coefficients, inclusion, residual_sd,
                    unscaled_innovation_sd, p00, p11):
        xdim = len(p00)
        model = boom.DynamicRegressionModel(xdim=xdim)
        for i in range(len(data)):
            model.add_data(data[i])

        sampler = boom.DynamicRegressionDirectGibbsSampler(
            model,
            1.0,
            1.0,
            boom.Vector(np.array([1.0] * xdim)),
            boom.Vector(np.array([1.0] * xdim)),
            boom.Vector(np.array([.25] * xdim)),
            boom.Vector(np.array([2.0] * xdim)),
            boom.Vector(np.array([1.0] * xdim)),
            boom.GlobalRng.rng)
        model.set_method(sampler)
        model.set_residual_sd(residual_sd)
        model.set_unscaled_innovation_sds(unscaled_innovation_sd)
        model.set_transition_probabilities(p00, p11)
        model.set_inclusion_indicators(inclusion)
        model.set_coefficients(coefficients)

        return model, sampler

    def test_data(self):
        time_point = boom.RegressionDataTimePoint()
        r1 = boom.RegressionData(1.0, np.random.randn(3))
        time_point.add_data(r1)
        self.assertEqual(3, time_point.xdim)
        self.assertEqual(1, time_point.sample_size)

        y = np.random.randn(10)
        x = np.random.randn(10, 3)
        for i in range(10):
            reg_data = boom.RegressionData(y[i], boom.Vector(x[i, :]))
            time_point.add_data(reg_data)

        self.assertEqual(11, time_point.sample_size)

    def model_smoke_test(self):
        xdim = 3
        typical_sample_size = 30
        time_dimension = 12
        model = boom.DynamicRegressionModel(xdim)

        data = self.simulate_null_data(
            time_dimension, typical_sample_size, xdim)

        for i in range(len(data)):
            model.add_data(data[i])

        sampler = boom.DynamicRegressionDirectGibbsSampler(
            model,
            1.0,
            1.0,
            boom.Vector(np.array([1.0] * xdim)),
            boom.Vector(np.array([1.0] * xdim)),
            boom.Vector(np.array([.25] * xdim)),
            boom.Vector(np.array([2.0] * xdim)),
            boom.Vector(np.array([1.0] * xdim)),
            boom.GlobalRng.rng
        )

        model.set_method(sampler)
        for i in range(10):
            model.sample_posterior()

    def test_draw_inclusion_indicators(self):
        # Make the coefficients big, so that effects are obvious.
        unscaled_innovation_sd = np.array([10, 20, 30])
        data, coefficients, inclusion = self.simulate_data_from_model(
            time_dimension=100,
            typical_sample_size=500,
            xdim=self._xdim,
            residual_sd=self._residual_sd,
            unscaled_innovation_sd=unscaled_innovation_sd,
            p00=self._p00,
            p11=self._p11)

        model, sampler = self.setup_model(
            data, coefficients, inclusion, self._residual_sd,
            unscaled_innovation_sd, self._p00, self._p11)

        niter = 1000
        draws = np.full((niter, model.xdim, model.time_dimension), -1)
        for i in range(niter):
            sampler.draw_inclusion_indicators()
            draws[i, :, :] = model.inclusion_indicators.to_numpy()

        posterior_mean = np.mean(draws[100:, :], axis=0)
        mean_vector = posterior_mean.flatten()
        inclusion_vector = inclusion.flatten()

        cor = R.corr(inclusion_vector, mean_vector)
        self.assertGreater(cor, .6)

    def test_draw_coefficients(self):
        # Make the coefficients big, so that effects are obvious.
        unscaled_innovation_sd = np.array([10, 20, 30])
        data, coefficients, inclusion = self.simulate_data_from_model(
            time_dimension=100,
            typical_sample_size=500,
            xdim=self._xdim,
            residual_sd=self._residual_sd,
            unscaled_innovation_sd=unscaled_innovation_sd,
            p00=self._p00,
            p11=self._p11)

        model, sampler = self.setup_model(
            data, coefficients, inclusion, self._residual_sd,
            unscaled_innovation_sd, self._p00, self._p11)

        niter = 1000
        draws = np.full((niter, model.xdim, model.time_dimension), np.NaN)

        for i in range(niter):
            model.draw_coefficients_given_inclusion(boom.GlobalRng.rng)
            draws[i, :, :] = model.all_coefficients.to_numpy()

        posterior_mean = np.mean(draws, axis=0)
        mean_vector = posterior_mean.flatten()
        beta_vector = coefficients.flatten()
        cor = R.corr(mean_vector, beta_vector)
        self.assertGreater(cor, .9)

    def test_draw_residual_variance(self):
        data, coefficients, inclusion = self.simulate_data_from_model(
            time_dimension=100,
            typical_sample_size=500,
            xdim=self._xdim,
            residual_sd=self._residual_sd,
            unscaled_innovation_sd=self._unscaled_innovation_sd,
            p00=self._p00,
            p11=self._p11)

        model, sampler = self.setup_model(
            data, coefficients, inclusion, self._residual_sd,
            self._unscaled_innovation_sd, self._p00, self._p11)

        niter = 1000
        draws = np.full(niter, -1)

        for i in range(niter):
            sampler.draw_residual_variance()
            draws[i] = model.residual_sd

        import pdb
        pdb.set_trace()
        print("Look at draws of the residual sd.")


    def test_draw_state_innovation_variance(self):
        pass

    def test_draw_transition_probabilities(self):
        pass

debug_mode_ = True

if debug_mode_:
    import pdb
    print("Hello, world!")
    rig = DynregTest()
    rig.setUp()
#     pdb.set_trace()
    rig.test_draw_inclusion_indicators()
    rig.test_draw_coefficients()
    rig.test_draw_residual_variance()
    print("All done!")

elif __name__ == "__main__":
    unittest.main()

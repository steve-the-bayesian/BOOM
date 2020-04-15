import unittest
import BayesBoom as boom
import numpy as np


class DynregTest(unittest.TestCase):
    def setUp(self):
        pass

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

    def test_model(self):
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

debug_mode_ = False

if debug_mode_:
#     import pdb
    rig = DynregTest()
    rig.setUp()
#     pdb.set_trace()
    rig.test_data()
    rig.test_model()

elif __name__ == "__main__":
    unittest.main()

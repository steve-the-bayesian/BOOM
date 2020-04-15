import unittest
import BayesBoom as boom
import numpy as np

class DynregTest(unittest.TestCase):
    def setUp(self):
        pass

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
        model = boom.DynamicRegressionModel(3)


        # add data

        # set posterior sampler

        #


if __name__ == "__main__":
    unittest.main()

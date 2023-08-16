import unittest
import BayesBoom.boom as boom
import BayesBoom.R as R
import numpy as np

class GaussianProcessRegressionTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_mean_prediction(self):
        nobs = 20
        X = np.random.randn(nobs, 1)
        residual_sd = 7.0
        intercept = 4
        y = intercept + 3 * X[:, 0] + np.random.randn(nobs) * residual_sd

        mean_function = boom.ZeroFunction()
        kernel = boom.MahalanobisKernel(
            R.to_boom_matrix(X),
            scale = 1.2,
            diagonal_shrinkage = .05
        )
        model = boom.GaussianProcessRegressionModel(
            mean_function=mean_function,
            kernel=kernel,
            residual_sd=residual_sd)

        model.add_data(R.to_boom_matrix(X), R.to_boom_vector(y))

        xnew = np.random.randn(5, 1)
        pred = model.predict_distribution(R.to_boom_matrix(xnew))

        self.assertIsInstance(pred, boom.MvnBase)


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

    rig = GaussianProcessRegressionTest()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_mean_prediction()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

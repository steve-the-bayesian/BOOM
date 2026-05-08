import unittest
import BayesBoom.R as R
import numpy as np
# import pandas as pd
import scipy.sparse


class TestLinearModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_lm(self):
        sample_size = 100
        xdim = 3
        X = np.random.randn(sample_size, xdim)
        X[:, 0] = 1.0
        beta = np.array([-3, 2, -1])
        yhat = X @ beta
        residual_sd = .2
        errors = np.random.randn(sample_size) * residual_sd
        y = yhat + errors

        model = R.lm(y, X)
        anova = model.anova_table
        print(anova)
        print(model.coefficient_table)
        for i in range(3):
            # The RNG should be be seeded (as it is in the setUp) method, if
            # this test is to be run in production.
            beta_hat = model.coefficient_table["coef"][i]
            se = model.coefficient_table["se"][i]
            self.assertLess((beta_hat - beta[i]) / se, 3)

        self.assertAlmostEqual(anova.SSE, np.sum(model.residuals**2))
        self.assertAlmostEqual(anova.SST, np.sum((y-np.mean(y))**2))
        self.assertGreaterEqual(model.Rsquare, 0.0)
        self.assertLessEqual(model.Rsquare, 1.0)

        true_residual_variance = residual_sd ** 2
        df = sample_size - xdim
        chisq_cutoff = scipy.stats.chi2.isf(.01, df)
        # df * residual_sd / true_residual_sd  ~ chisq_df
        # so we have residual_sd ~ chisq_df * true_residual_variance / df
        self.assertLess(model.residual_variance,
                        chisq_cutoff * true_residual_variance / df)


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestLinearModel()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_lm()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

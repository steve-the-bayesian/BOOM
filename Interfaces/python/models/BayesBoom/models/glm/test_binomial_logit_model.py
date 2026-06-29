import unittest
import numpy as np
import BayesBoom.boom as boom

from .binomial_logit_model import (
    BinomialLogitModel,
    BinomialLogitMvnPrior,
    BinomialLogitSpikeSlabPrior,
    LogitZellnerPrior,
)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_logistic_data(n, beta, seed=42):
    """Return (X, y) for binary logistic regression with an intercept column."""
    rng = np.random.RandomState(seed)
    beta = np.asarray(beta, dtype=float)
    xdim = len(beta)
    X = np.column_stack([np.ones(n), rng.randn(n, xdim - 1)])
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = rng.binomial(1, p).astype(float)
    return X, y


def simulate_binomial_data(n, beta, trials_per_obs=20, seed=42):
    """Return (X, y, trials) for grouped binomial data with an intercept column."""
    rng = np.random.RandomState(seed)
    beta = np.asarray(beta, dtype=float)
    xdim = len(beta)
    X = np.column_stack([np.ones(n), rng.randn(n, xdim - 1)])
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    trials = np.full(n, float(trials_per_obs))
    y = rng.binomial(trials_per_obs, p).astype(float)
    return X, y, trials


def run_mcmc(model, niter=1000, burnin=500):
    """Burn in then collect `niter` coefficient draws from `model`."""
    model.boom()
    for _ in range(burnin):
        model.sample_posterior()
    draws = np.empty((niter, model.xdim))
    for i in range(niter):
        model.sample_posterior()
        draws[i] = model.coefficients
    return draws


# ---------------------------------------------------------------------------
# Construction and basic interface
# ---------------------------------------------------------------------------

class TestBinomialLogitModelConstruction(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        boom.GlobalRng.rng.seed(8675309)

    def test_binary_data_shapes(self):
        X, y = simulate_logistic_data(100, [0.5, -1.0, 0.8])
        model = BinomialLogitModel(X, y)
        self.assertEqual(model.xdim, 3)
        self.assertEqual(model.sample_size, 100)

    def test_binomial_grouped_data_shapes(self):
        X, y, trials = simulate_binomial_data(50, [0.0, 1.0])
        model = BinomialLogitModel(X, y, trials=trials)
        self.assertEqual(model.xdim, 2)
        self.assertEqual(model.sample_size, 50)

    def test_1d_predictor_is_reshaped(self):
        x = np.random.randn(30)
        y = np.random.randint(0, 2, 30).astype(float)
        model = BinomialLogitModel(x, y)
        self.assertEqual(model.xdim, 1)

    def test_y_x_length_mismatch_raises(self):
        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 8).astype(float)
        with self.assertRaises(ValueError):
            BinomialLogitModel(X, y)

    def test_trials_length_mismatch_raises(self):
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 10).astype(float)
        with self.assertRaises(ValueError):
            BinomialLogitModel(X, y, trials=np.ones(8))

    def test_boom_returns_boom_binomial_logit_model(self):
        import BayesBoom.boom as boom
        X, y = simulate_logistic_data(50, [0.0, 1.0])
        model = BinomialLogitModel(X, y)
        self.assertIsInstance(model.boom(), boom.BinomialLogitModel)

    def test_boom_is_cached(self):
        X, y = simulate_logistic_data(50, [0.0, 1.0])
        model = BinomialLogitModel(X, y)
        self.assertIs(model.boom(), model.boom())

    def test_default_prior_is_mvn(self):
        X, y = simulate_logistic_data(50, [0.0, 1.0])
        model = BinomialLogitModel(X, y)
        model.boom()
        self.assertIsInstance(model.prior, BinomialLogitMvnPrior)

    def test_coefficients_shape(self):
        X, y = simulate_logistic_data(50, [0.0, 1.0, -0.5])
        model = BinomialLogitModel(X, y)
        model.sample_posterior()
        self.assertEqual(model.coefficients.shape, (3,))


# ---------------------------------------------------------------------------
# add_data
# ---------------------------------------------------------------------------

class TestAddData(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_add_data_before_boom_updates_sample_size(self):
        X, y = simulate_logistic_data(50, [0.5, -0.5])
        model = BinomialLogitModel(X[:30], y[:30])
        model.add_data(y[30:], X[30:])
        self.assertEqual(model.sample_size, 50)

    def test_add_data_after_boom_updates_sample_size(self):
        X, y = simulate_logistic_data(50, [0.5, -0.5])
        model = BinomialLogitModel(X[:30], y[:30])
        model.boom()
        model.add_data(y[30:], X[30:])
        self.assertEqual(model.sample_size, 50)

    def test_add_binomial_data_with_trials(self):
        X, y, trials = simulate_binomial_data(30, [0.0, 1.0])
        model = BinomialLogitModel(X[:20], y[:20], trials=trials[:20])
        model.add_data(y[20:], X[20:], trials=trials[20:])
        self.assertEqual(model.sample_size, 30)

    def test_add_data_default_trials_are_ones(self):
        X, y = simulate_logistic_data(40, [0.0, 1.0])
        model = BinomialLogitModel(X[:20], y[:20])
        model.add_data(y[20:], X[20:])
        np.testing.assert_array_equal(model._trials, np.ones(40))


# ---------------------------------------------------------------------------
# MCMC with BinomialLogitMvnPrior
# ---------------------------------------------------------------------------

class TestMcmcMvnPrior(unittest.TestCase):
    """MCMC with BinomialLogitMvnPrior (default auxmix sampler)."""

    TRUE_BETA = np.array([0.5, 1.5, -1.0])
    N = 500
    NITER = 1000
    BURNIN = 500
    ATOL = 0.35

    def setUp(self):
        np.random.seed(8675309)
        boom.GlobalRng.rng.seed(8675309)
        self._X, self._y = simulate_logistic_data(self.N, self.TRUE_BETA)

    def test_default_prior_recovers_true_beta(self):
        model = BinomialLogitModel(self._X, self._y)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        np.testing.assert_allclose(
            draws.mean(axis=0), self.TRUE_BETA, atol=self.ATOL)

    def test_explicit_variance_scale(self):
        prior = BinomialLogitMvnPrior(variance_scale=10.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        np.testing.assert_allclose(
            draws.mean(axis=0), self.TRUE_BETA, atol=self.ATOL)

    def test_explicit_mu_and_Sigma(self):
        xdim = self._X.shape[1]
        prior = BinomialLogitMvnPrior(
            mu=np.zeros(xdim), Sigma=np.eye(xdim) * 10.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        np.testing.assert_allclose(
            draws.mean(axis=0), self.TRUE_BETA, atol=self.ATOL)

    def test_grouped_binomial_data(self):
        X, y, trials = simulate_binomial_data(200, self.TRUE_BETA)
        model = BinomialLogitModel(X, y, trials=trials)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        np.testing.assert_allclose(
            draws.mean(axis=0), self.TRUE_BETA, atol=self.ATOL)

    def test_clt_threshold_parameter(self):
        prior = BinomialLogitMvnPrior(clt_threshold=5)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        model.sample_posterior()  # no exception

    def test_draws_are_finite(self):
        model = BinomialLogitModel(self._X, self._y)
        draws = run_mcmc(model, niter=200, burnin=100)
        self.assertTrue(np.all(np.isfinite(draws)))


# ---------------------------------------------------------------------------
# MCMC with BinomialLogitSpikeSlabPrior
# ---------------------------------------------------------------------------

class TestMcmcSpikeSlabPrior(unittest.TestCase):
    """MCMC with BinomialLogitSpikeSlabPrior.

    TRUE_BETA has two non-zero entries (indices 0, 1, 4) and two zero
    entries (indices 2, 3).  After MCMC the posterior inclusion
    probabilities should reflect this sparsity pattern.

    In BOOM's spike-and-slab framework, excluded coefficients are set to
    exactly 0.0 in each draw, so we estimate the posterior inclusion
    probability as the fraction of draws in which the coefficient is
    non-zero.
    """

    TRUE_BETA = np.array([0.5, 2.0, 0.0, 0.0, -1.5])
    N = 600
    NITER = 2000
    BURNIN = 500

    def setUp(self):
        np.random.seed(8675309)
        boom.GlobalRng.rng.seed(8675309)
        self._X, self._y = simulate_logistic_data(self.N, self.TRUE_BETA)

    def _posterior_inclusion(self, draws):
        """Fraction of draws where each coefficient is non-zero."""
        return (draws != 0.0).mean(axis=0)

    def test_spike_slab_prior_accepted(self):
        prior = BinomialLogitSpikeSlabPrior(
            variance_scale=10.0, expected_model_size=3.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        model.sample_posterior()  # no exception

    def test_inclusion_probability_high_for_signal(self):
        prior = BinomialLogitSpikeSlabPrior(
            variance_scale=10.0, expected_model_size=3.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        post_incl = self._posterior_inclusion(draws)
        for j in [0, 1, 4]:  # non-zero coefficients
            self.assertGreater(
                post_incl[j], 0.5,
                msg=f"Coef[{j}]: expected high inclusion, got {post_incl[j]:.3f}")

    def test_inclusion_probability_low_for_noise(self):
        prior = BinomialLogitSpikeSlabPrior(
            variance_scale=10.0, expected_model_size=3.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        post_incl = self._posterior_inclusion(draws)
        for j in [2, 3]:  # zero coefficients
            self.assertLess(
                post_incl[j], 0.5,
                msg=f"Coef[{j}]: expected low inclusion, got {post_incl[j]:.3f}")

    def test_posterior_mean_of_included_draws_near_truth(self):
        prior = BinomialLogitSpikeSlabPrior(
            variance_scale=10.0, expected_model_size=3.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        for j, true_val in enumerate(self.TRUE_BETA):
            if true_val == 0.0:
                continue
            included = draws[:, j] != 0.0
            if included.sum() < 50:
                continue  # not enough included draws to test
            post_mean = draws[included, j].mean()
            self.assertAlmostEqual(
                post_mean, true_val, delta=0.5,
                msg=f"Coef[{j}]: post mean={post_mean:.3f}, truth={true_val}")

    def test_custom_mu_and_Sigma(self):
        xdim = self._X.shape[1]
        prior = BinomialLogitSpikeSlabPrior(
            mu=np.zeros(xdim),
            Sigma=np.eye(xdim) * 5.0,
            expected_model_size=2.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        model.sample_posterior()
        self.assertEqual(model.coefficients.shape, (xdim,))

    def test_clt_threshold_parameter(self):
        prior = BinomialLogitSpikeSlabPrior(clt_threshold=3)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        model.sample_posterior()  # no exception

    def test_grouped_binomial_data(self):
        """Spike-and-slab also works with grouped binomial observations."""
        X, y, trials = simulate_binomial_data(300, self.TRUE_BETA)
        prior = BinomialLogitSpikeSlabPrior(
            variance_scale=10.0, expected_model_size=3.0)
        model = BinomialLogitModel(X, y, trials=trials, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        post_incl = self._posterior_inclusion(draws)
        for j in [0, 1, 4]:
            self.assertGreater(post_incl[j], 0.5)

    def test_draws_are_finite(self):
        prior = BinomialLogitSpikeSlabPrior(expected_model_size=3.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=200, burnin=100)
        self.assertTrue(np.all(np.isfinite(draws)))

    def test_expected_model_size_one(self):
        """Very sparse prior: most coefficients should be excluded."""
        prior = BinomialLogitSpikeSlabPrior(
            variance_scale=10.0, expected_model_size=1.0)
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        # Null predictors should still be excluded more than signal predictors.
        post_incl = self._posterior_inclusion(draws)
        self.assertGreater(post_incl[1], post_incl[2])
        self.assertGreater(post_incl[4], post_incl[3])


# ---------------------------------------------------------------------------
# LogitZellnerPrior
# ---------------------------------------------------------------------------

class TestLogitZellnerPrior(unittest.TestCase):
    """Tests for LogitZellnerPrior (Zellner g-prior style variable selection)."""

    TRUE_BETA = np.array([0.5, 1.5, 0.0, -1.0])
    N = 600
    NITER = 2000
    BURNIN = 500

    def setUp(self):
        np.random.seed(8675309)
        boom.GlobalRng.rng.seed(8675309)
        self._X, self._y = simulate_logistic_data(self.N, self.TRUE_BETA)

    def _posterior_inclusion(self, draws):
        return (draws != 0.0).mean(axis=0)

    def test_construction_from_binary_data(self):
        prior = LogitZellnerPrior(
            predictors=self._X,
            successes=self._y,
        )
        self.assertIsNotNone(prior)
        self.assertEqual(prior._mean.shape, (4,))
        self.assertEqual(prior._precision.shape, (4, 4))

    def test_construction_with_trials(self):
        X, y, trials = simulate_binomial_data(self.N, self.TRUE_BETA)
        prior = LogitZellnerPrior(
            predictors=X,
            successes=y,
            trials=trials,
        )
        self.assertIsNotNone(prior)

    def test_construction_no_successes_uses_prior_prob(self):
        prior = LogitZellnerPrior(
            predictors=self._X,
            prior_success_probability=0.3,
        )
        # Intercept prior mean should be near logit(0.3)
        expected_intercept = np.log(0.3 / 0.7)
        self.assertAlmostEqual(prior._mean[0], expected_intercept, places=5)

    def test_from_parameters(self):
        xdim = 4
        mean = np.array([0.1, 0.2, 0.3, 0.4])
        precision = np.eye(xdim) * 2.0
        incl_probs = np.full(xdim, 0.5)
        prior = LogitZellnerPrior.from_parameters(mean, precision, incl_probs)
        np.testing.assert_array_equal(prior._mean, mean)
        np.testing.assert_array_equal(
            prior._prior_inclusion_probabilities, incl_probs)

    def test_expected_model_size_sets_inclusion_probs(self):
        xdim = self._X.shape[1]
        ems = 2.0
        prior = LogitZellnerPrior(
            predictors=self._X,
            successes=self._y,
            expected_model_size=ems,
        )
        expected_prob = ems / xdim
        np.testing.assert_allclose(
            prior._prior_inclusion_probabilities,
            np.full(xdim, expected_prob))

    def test_mcmc_recovers_signal_coefficients(self):
        prior = LogitZellnerPrior(
            predictors=self._X,
            successes=self._y,
            expected_model_size=1.5,
            prior_information_weight=0.1,
        )
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        post_incl = self._posterior_inclusion(draws)
        for j in [0, 1, 3]:  # non-zero coefficients
            self.assertGreater(
                post_incl[j], 0.5,
                msg=f"Coef[{j}]: expected high inclusion, got {post_incl[j]:.3f}")

    def test_mcmc_excludes_null_coefficients(self):
        # prior_information_weight=0.1 gives a diffuse slab (variance ~10),
        # which gives the spike-slab sampler enough contrast to exclude null
        # coefficients.  expected_model_size=1.5 with xdim=4 sets the prior
        # inclusion probability to 0.375, providing a slight lean toward
        # sparsity so that a neutral Bayes factor reliably yields PIP < 0.5.
        prior = LogitZellnerPrior(
            predictors=self._X,
            successes=self._y,
            expected_model_size=1.5,
            prior_information_weight=0.1,
        )
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        post_incl = self._posterior_inclusion(draws)
        self.assertLess(
            post_incl[2], 0.5,
            msg=f"Coef[2]: expected low inclusion, got {post_incl[2]:.3f}")

    def test_mcmc_posterior_mean_near_truth(self):
        prior = LogitZellnerPrior(
            predictors=self._X,
            successes=self._y,
            expected_model_size=1.5,
            prior_information_weight=0.1,
        )
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=self.NITER, burnin=self.BURNIN)
        for j, true_val in enumerate(self.TRUE_BETA):
            if true_val == 0.0:
                continue
            included = draws[:, j] != 0.0
            if included.sum() < 50:
                continue
            post_mean = draws[included, j].mean()
            self.assertAlmostEqual(
                post_mean, true_val, delta=0.5,
                msg=f"Coef[{j}]: post mean={post_mean:.3f}, truth={true_val}")

    def test_max_flips_limits_model_moves(self):
        prior = LogitZellnerPrior(
            predictors=self._X,
            successes=self._y,
            max_flips=1,
        )
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        model.sample_posterior()  # no exception

    def test_draws_are_finite(self):
        prior = LogitZellnerPrior(
            predictors=self._X,
            successes=self._y,
        )
        model = BinomialLogitModel(self._X, self._y, prior=prior)
        draws = run_mcmc(model, niter=200, burnin=100)
        self.assertTrue(np.all(np.isfinite(draws)))


# ---------------------------------------------------------------------------
# Debug entry point
# ---------------------------------------------------------------------------

_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    print("Hello, world!")

    rig = TestMcmcSpikeSlabPrior()
    rig.setUp()
    rig.test_inclusion_probability_high_for_signal()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

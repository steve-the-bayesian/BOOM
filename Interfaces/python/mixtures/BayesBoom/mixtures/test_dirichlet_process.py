import unittest
import BayesBoom.mixtures as mix
import numpy as np
import BayesBoom.R as R


class TestLabelSwitching(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_permutations(self):
        mu1 = np.zeros(3)
        mu2 = np.ones(3) * 4
        # mu2 = np.array([1, 10, 100])

        Sigma1 = np.diag(np.ones(3))
        Sigma2 = np.diag(np.ones(3))

        y1 = np.random.randn(100, 3) + mu1
        y2 = np.random.randn(150, 3) + mu2
        y = np.concatenate((y1, y2), axis=0)

        model = mix.DirichletProcessMvn(y)
        model.mcmc(100, ping=-1, permute_state_labels=False)

        true_cluster_probs1 = R.dmvn(y, mu1, Sigma1) * 100 / 250
        true_cluster_probs2 = R.dmvn(y, mu2, Sigma2) * 150 / 250
        total = true_cluster_probs1 + true_cluster_probs2

        true_cluster_probs = R.cbind(true_cluster_probs1 / total,
                                     true_cluster_probs2 / total)

        niter = 10
        nobs = y.shape[0]
        draws = np.empty((niter, nobs))
        perm = np.empty(niter)

        for i in range(niter):
            u = np.random.rand(nobs)
            draws[i, :] = u < true_cluster_probs[:, 1]
            v = np.random.rand(1)
            if v < .5:
                perm[i] = 1
                draws[i, :] = 1 - draws[i, :]
            else:
                perm[i] = 0

        fitted_perm = mix.identify_permutation_from_labels(draws)
        self.assertTrue(np.allclose(perm, fitted_perm[:, 1]))

    def test_choose_permutation(self):
        n1 = 100
        n2 = 200
        mu1 = np.zeros(3)
        mu2 = np.ones(3) * 4

        Sigma1 = np.diag(np.ones(3))
        Sigma2 = np.diag(np.ones(3))

        y1 = R.rmvn(n1, mu1, Sigma1)
        y2 = R.rmvn(n2, mu2, Sigma2)
        y = np.concatenate((y1, y2), axis=0)

        model = mix.DirichletProcessMvn(y)
        model.mcmc(1000, ping=-1, permute_state_labels=True)

        nscore1 = 3
        nscore2 = 5
        data_to_cluster = np.concatenate(
            (R.rmvn(nscore1, mu1, Sigma1),
             R.rmvn(nscore2, mu2, Sigma2)), axis=0)

        clusters = model.cluster(data_to_cluster, nclusters=2, prob=False)
        cluster1 = clusters[0]
        self.assertTrue(np.all(clusters[:nscore1] == cluster1))
        cluster2 = clusters[-1]
        self.assertTrue(np.all(clusters[nscore1:] == cluster2))

        cluster_probs = model.cluster(data_to_cluster, nclusters=2, prob=True)
        self.assertTrue(np.all(
            cluster_probs[:nscore1, cluster1] > cluster_probs[
                :nscore1, cluster2]))
        self.assertTrue(np.all(
            cluster_probs[nscore1:, cluster2] > cluster_probs[
                nscore1:, cluster1]))


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestLabelSwitching()

    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_choose_permutation()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

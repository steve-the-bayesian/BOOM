import unittest
import BayesBoom.R as R
import numpy as np
import scipy.stats as ss


class TestAssignClasses(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def simulate_data(self,
                      sample_size: int,
                      class_probs: np.ndarray,
                      binomial_n: int,
                      binomial_probs: np.ndarray):
        true_classes = np.random.choice(
            range(len(class_probs)),
            size=sample_size,
            replace=True,
            p=class_probs)

        y = np.random.binomial(binomial_n,
                               binomial_probs[true_classes],
                               size=sample_size)

        posterior = np.array([class_probs.tolist()]
                             * sample_size)
        likelihood = np.empty_like(posterior)
        for k in range(likelihood.shape[1]):
            likelihood[:, k] = ss.binom.pmf(y, binomial_n, binomial_probs[k])
        posterior *= likelihood
        normalizing_constant = posterior.sum(axis=1).reshape((-1, 1))
        posterior = posterior / normalizing_constant

        self._posterior = posterior
        self._class_probs = class_probs
        self._true_classes = true_classes

    def test_assignment_weak_case(self):
        self.simulate_data(sample_size=1000,
                           class_probs=np.array([.3, .4, .3]),
                           binomial_n=1,
                           binomial_probs=np.array([.4, .45, .5]))
        classes = R.assign_classes(
            self._posterior,
            self._class_probs)
        self.assertIsInstance(classes, list)

        assigner = R.ClassAssigner()
        classes2 = assigner.assign(self._posterior,
                                   self._class_probs)
        self.assertLess(assigner.kl, .10)

        freq1 = R.table(classes)
        freq1 = freq1 / np.sum(freq1)
        freq2 = R.table(classes2)
        freq2 = freq2 / np.sum(freq2)

        kl = R.kl_divergence(freq1, freq2)
        self.assertLess(kl, .1)

    def test_assignment_strong_case(self):
        self.simulate_data(sample_size=1000,
                           class_probs=np.array([.3, .4, .3]),
                           binomial_n=100,
                           binomial_probs=np.array([.2, .5, .8]))
        classes = np.array(R.assign_classes(
            self._posterior,
            self._class_probs))

        assigner = R.ClassAssigner()
        classes2 = np.array(assigner.assign(self._posterior,
                                            self._class_probs))

        truth = np.array(self._true_classes)
        num_mistakes1 = np.sum(classes != truth)
        num_mistakes2 = np.sum(classes2 != truth)
        self.assertLess(num_mistakes1, 10)
        self.assertLess(num_mistakes2, 10)


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestAssignClasses()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_assignment_weak_case()
    rig.test_assignment_strong_case()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

import BayesBoom.impute as impute
import BayesBoom.test_utils as tu

import unittest
import numpy as np


class ImputerTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self.data = tu.simulate_data(
            sample_size=100, numeric_dim=3, cat_levels={
                "colors": ["red", "blue", "green"],
                "states": ["CA", "TX", "NY", "OR", "MA"],
            })
        self.model = impute.MixedDataImputer()

    def test_empty_model(self):
        self.assertEqual(self.model.nclusters, 0)
        self.assertEqual(self.model.niter, 0)

        self.model.find_atoms(self.data)
        self.assertEqual(self.model._numeric_colnames, ['X1', 'X2', 'X3'])
        self.assertTrue(isinstance(self.model._atoms, dict))
        self.assertEqual(self.model._atoms["X1"], [])

    def test_fit_imputation_model(self):
        data = self.data
        data["X1"][:10] = np.NaN
        data["X2"][5:15] = np.NaN
        self.model.find_atoms(data)
        num_clusters = 5
        niter = 100
        self.model.train_model(data=data, niter=niter,
                               nclusters=num_clusters)
        self.assertEqual(num_clusters, self.model.nclusters)
        num_states = 5
        num_colors = 3
        xdim = 1 + (num_colors - 1) + (num_states - 1)
        self.assertEqual(self.model.xdim, xdim)
        self.assertEqual(3, self.model.ydim)
        self.assertEqual(self.model.atom_probs["X1"].shape,
                         (niter, num_clusters, 1))


_debug_mode = True

if _debug_mode:
    import pdb
    rig = ImputerTest()
    rig.setUp()
    rig.test_fit_imputation_model()

elif __name__ == "__main__":
    unittest.main()

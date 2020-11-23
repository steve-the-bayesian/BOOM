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

        self.model = impute.MissingDataImputer()

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
        self.model.train_model(data=data, niter=100, num_clusters=3)

_debug_mode = True

if _debug_mode:
    import pdb
    rig = ImputerTest()
    rig.setUp()
    rig.test_fit_imputation_model()

elif __name__ == "__main__":
    unittest.main()

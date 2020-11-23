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

    def test_stuff(self):
        self.assertEqual(self.model.nclusters, 0)


if __name__ == "__main__":
    unittest.main()

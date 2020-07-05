import unittest
import R
import numpy as np
import pandas as pd
import BayesBoom as boom
import pdb


class DataTableTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        boom.GlobalRng.rng.seed(8675309)
        n = 100
        numerics = np.random.randn(n, 3)
        colors = np.random.choice(
            ["red", "blue", "green"], size=n, replace=True)
        shapes = np.random.choice(
            ["circle", "square", "triangle"], size=n, replace=True)
        self._data = pd.DataFrame({
                "X1": numerics[:, 0],
                "colors": colors,
                "X2": numerics[:, 1],
                "shapes": shapes,
                "X3": numerics[:, 2]
            })

    def test_data_table(self):
        table = R.create_data_table(self._data)
        self.assertEqual(table.nrow, self._data.shape[0])
        self.assertEqual(table.ncol, self._data.shape[1])

    def test_autoclean_construction(self):
        ac = R.AutoClean()
        ac.train_model(self._data, nclusters=3, niter=10)
        print("all done!")

# unittest.main()
rig = DataTableTest()
rig.setUp()
rig.test_autoclean_construction()

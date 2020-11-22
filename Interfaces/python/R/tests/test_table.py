import unittest
import numpy as np
from BayesBoom.R import table


class TableTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        sample_size = 100
        regions = ["North", "South", "East", "West"]
        self.regions = np.random.choice(regions, sample_size, replace=True)
        colors = ["red", "blue", "green"]
        self.colors = np.random.choice(colors, sample_size, replace=True)

    def test_univariate(self):
        tab = table(self.regions)
        self.assertEqual(len(tab), 4)
        self.assertEqual(np.sum(tab), 100)

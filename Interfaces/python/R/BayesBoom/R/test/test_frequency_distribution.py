import unittest
import BayesBoom.R as R
import BayesBoom.test_utils as test_utils
import numpy as np
import pandas as pd
import scipy.sparse


class TestFrequencyDistribution(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        self._data = test_utils.simulate_data(
            sample_size=100,
            cat_levels={"stooges": ["Larry", "Moe", "Curly", "Shemp"]}
        )

    def test_without_nans(self):
        stooges = self._data["stooges"]
        dist = R.FrequencyDistribution(stooges)

        counts = stooges.value_counts()

        numpy_dist = R.FrequencyDistribution(np.array(stooges))
        list_dist = R.FrequencyDistribution(stooges.tolist())
        self.assertEqual(dist.sample_size, self._data.shape[0])
        self.assertEqual(list_dist.sample_size, self._data.shape[0])
        self.assertEqual(numpy_dist.sample_size, self._data.shape[0])

        self.assertEqual(dist.counts["Larry"], counts["Larry"])
        self.assertEqual(dist.counts["Moe"], counts["Moe"])
        self.assertEqual(dist.counts["Curly"], counts["Curly"])
        self.assertEqual(dist.counts["Shemp"], counts["Shemp"])

        self.assertTrue(np.allclose(dist.counts, list_dist.counts))
        self.assertTrue(np.allclose(dist.counts, numpy_dist.counts))

        self.assertEqual(dist.nan_count, 0)
        self.assertEqual(list_dist.nan_count, 0)
        self.assertEqual(numpy_dist.nan_count, 0)
        
    def test_without_nans_with_categories(self):
        stooges = self._data["stooges"]
        categories = ["Larry", "Moe", "Curly", "Shemp", "Frank"]

        dist = R.FrequencyDistribution(stooges, categories=categories)
        counts = stooges.value_counts()

        self.assertEqual(dist.sample_size, self._data.shape[0])
        self.assertEqual(dist.counts["Larry"], counts["Larry"])
        self.assertEqual(dist.counts["Moe"], counts["Moe"])
        self.assertEqual(dist.counts["Curly"], counts["Curly"])
        self.assertEqual(dist.counts["Shemp"], counts["Shemp"])
        self.assertEqual(dist.counts["Frank"], 0)

        list_dist = R.FrequencyDistribution(stooges.tolist(), categories=categories)
        numpy_dist = R.FrequencyDistribution(np.array(stooges), categories=categories)

        self.assertTrue(np.allclose(list_dist.counts, dist.counts))
        self.assertTrue(np.allclose(numpy_dist.counts, dist.counts))

        self.assertEqual(dist.nan_count, 0)
        self.assertEqual(list_dist.nan_count, 0)
        self.assertEqual(numpy_dist.nan_count, 0)

    def test_with_nans(self):
        stooges = self._data["stooges"].copy()
        stooges[0] = np.nan
        stooges[3] = np.nan
        stooges[17] = np.nan
        
        dist = R.FrequencyDistribution(stooges)

        counts = stooges.value_counts()
        numpy_dist = R.FrequencyDistribution(np.array(stooges))
        list_dist = R.FrequencyDistribution(stooges.tolist())
        
        self.assertEqual(dist.sample_size, self._data.shape[0])
        self.assertEqual(list_dist.sample_size, self._data.shape[0])
        self.assertEqual(numpy_dist.sample_size, self._data.shape[0])

        self.assertEqual(dist.counts["Larry"], counts["Larry"])
        self.assertEqual(dist.counts["Moe"], counts["Moe"])
        self.assertEqual(dist.counts["Curly"], counts["Curly"])
        self.assertEqual(dist.counts["Shemp"], counts["Shemp"])

        self.assertEqual(dist.nan_count, 3)
        self.assertEqual(list_dist.nan_count, 3)
        self.assertEqual(numpy_dist.nan_count, 3)

    def test_with_nans_and_categories(self):
        stooges = self._data["stooges"].copy()
        stooges[0] = np.nan
        stooges[3] = np.nan
        stooges[17] = np.nan

        categories = ["Larry", "Moe", "Curly", "Shemp", "Frank"]        
        dist = R.FrequencyDistribution(stooges, categories=categories)

        counts = stooges.value_counts()
        numpy_dist = R.FrequencyDistribution(np.array(stooges),
                                             categories=categories)
        list_dist = R.FrequencyDistribution(stooges.tolist(),
                                            categories=categories)
        
        self.assertEqual(dist.sample_size, self._data.shape[0])
        self.assertEqual(list_dist.sample_size, self._data.shape[0])
        self.assertEqual(numpy_dist.sample_size, self._data.shape[0])

        self.assertEqual(dist.counts["Larry"], counts["Larry"])
        self.assertEqual(dist.counts["Moe"], counts["Moe"])
        self.assertEqual(dist.counts["Curly"], counts["Curly"])
        self.assertEqual(dist.counts["Shemp"], counts["Shemp"])
        self.assertEqual(dist.counts["Frank"], 0)

        self.assertEqual(dist.nan_count, 3)
        self.assertEqual(list_dist.nan_count, 3)
        self.assertEqual(numpy_dist.nan_count, 3)


    def test_collapse(self):
        data = (
            ["red"] * 10
            + ["orange"] * 11
            + ["yellow"] * 8
            + ["green"] * 9
            + ["blue"] * 5
            + ["indigo"] * 3
            + ["violet"]
        )
        sample_size = len(data)
        dist = R.FrequencyDistribution(data)
        self.assertEqual(dist.sample_size, sample_size)
        dist.collapse(5, '[Other]')
        self.assertEqual(dist.sample_size, sample_size)
        self.assertEqual(dist["red"], 10)
        self.assertEqual(dist["yellow"], 8)
        self.assertEqual(dist["green"], 9)
        self.assertEqual(dist["[Other]"], 4)
        

_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestFrequencyDistribution()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_without_nans()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

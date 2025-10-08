import unittest
import BayesBoom.R as R
import numpy as np
import pandas as pd

import BayesBoom.test_utils as test_utils


class TestSummary(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

        self._data = test_utils.simulate_data(
            sample_size=1000,
            numeric_dim=1,
            cat_levels={
                "color": ["red", "blue", "green"],
                "stooges": ["Larry", "Moe", "Curly", "Shemp"],
                "binary": ["Yes", "No"],
            },
            date_fields={
                "birthday": ("1970-01-01", "2020-08-13"),
                "anniversary": ("1988-03-12", "2020-01-29"),
            },
            high_cardinality_fields={
                "zip_code": 3,
                "address": 12,
            },
        )

    def test_numeric(self):
        numeric = R.NumericSummary(self._data["X1"])
        sample_size = self._data.shape[0]
        self.assertEqual(numeric.sample_size, sample_size)

        y = self._data["X1"].copy()
        y[3] = np.nan
        y[12] = np.nan

        numeric = R.NumericSummary(y)
        self.assertEqual(numeric.sample_size, sample_size)
        self.assertEqual(numeric.number_missing, 2)
        self.assertEqual(numeric.number_observed, sample_size - 2)

        self.assertAlmostEqual(numeric.proportion_missing,
                               2.0 / sample_size)

        self.assertFalse(numeric.missing_values_assumed)
        print(numeric.potential_missing_value_codes)

    def test_categorical(self):
        cat = R.CategoricalSummary(self._data["color"])
        self.assertEqual(cat.sample_size, self._data.shape[0])
        self.assertFalse(cat.is_binary)

        bin = R.CategoricalSummary(self._data["binary"])
        self.assertTrue(bin.is_binary)

    def test_datetime(self):
        bday = self._data["birthday"].copy()
        sample_size = self._data.shape[0]
        bsum = R.DateTimeSummary(bday)
        self.assertEqual(sample_size, bsum.sample_size)
        self.assertEqual(sample_size, bsum.number_observed)
        self.assertEqual(0, bsum.number_missing)

        bday[3] = np.nan
        bday[5] = np.nan
        bsum = R.DateTimeSummary(bday)
        self.assertEqual(bsum.sample_size, sample_size)
        self.assertEqual(bsum.number_missing, 2)
        self.assertEqual(bsum.number_observed, sample_size - 2)
        
    def test_high_cardinality(self):
        zip_summary = R.HighCardinalitySummary(self._data["zip_code"])
        sample_size = self._data.shape[0]
        
        self.assertEqual(zip_summary.sample_size, sample_size)
        zipcodes = self._data["zip_code"].copy()
        zipcodes[3] = np.nan
        zipcodes[5] = np.nan

        zip_summary = R.HighCardinalitySummary(zipcodes)
        self.assertEqual(zip_summary.number_missing, 2)
        self.assertEqual(zip_summary.number_observed, sample_size - 2)
                         
        self.assertEqual(zip_summary.sample_size, sample_size)

    def test_correct_summary_type(self):
        summaries = R.summary(self._data)
        self.assertEqual(len(summaries), self._data.shape[1])
        self.assertIsInstance(summaries, dict)
        
        self.assertIsInstance(summaries["X1"], R.NumericSummary)
        self.assertIsInstance(summaries["color"], R.CategoricalSummary)
        self.assertIsInstance(summaries["stooges"], R.CategoricalSummary)
        self.assertIsInstance(summaries["birthday"], R.DateTimeSummary)
        self.assertIsInstance(summaries["anniversary"], R.DateTimeSummary)
        self.assertIsInstance(summaries["zip_code"], R.HighCardinalitySummary)
        self.assertIsInstance(summaries["address"], R.HighCardinalitySummary)
        

_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestSummary()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_correct_summary_type()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

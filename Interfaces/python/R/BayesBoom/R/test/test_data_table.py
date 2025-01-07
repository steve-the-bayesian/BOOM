import unittest
import BayesBoom.R as R
import numpy as np
import pandas as pd
import BayesBoom.boom as boom

# pylint: disable=unused-import
import pdb


def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2) + 1):
        yield chr(c)


# Some categorical data to use for testing.
LETTERS = list(char_range("a", "z"))

# fmt: off
us_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
             "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
             "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
             "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
             "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
# fmt: on


def random_words(sample_size, word_length=10):
    np.random.seed(8675309)
    return ["".join(np.random.choice(LETTERS, size=word_length))
            for i in range(sample_size)]


class DataTableTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)
        boom.GlobalRng.rng.seed(8675309)

        sample_size = 100
        numerics = np.random.randn(sample_size, 3)
        colors = np.random.choice(
            ["red", "blue", "green"], size=sample_size, replace=True)
        shapes = np.random.choice(
            ["circle", "square", "triangle"], size=sample_size, replace=True)
        self._data = pd.DataFrame({
                "X1": numerics[:, 0],
                "colors": colors,
                "X2": numerics[:, 1],
                "shapes": shapes,
                "X3": numerics[:, 2]
            })

    def create_base_dataset(
            self, sample_size, num_numeric, num_cat, num_levels):
        xdim = 1 + num_cat * (num_levels - 1)
        cats = {}
        levels = {}
        encoders = []
        for i in range(num_cat):
            # local_levels are the levels this variable can assume.
            local_levels = random_words(num_levels)
            vname = "cat" + str(i+1)
            values = np.random.choice(local_levels, sample_size)
            cats[vname] = values
            levels[vname] = local_levels
            encoders.append(boom.EffectsEncoder(i, local_levels))

        ydim = num_numeric
        self._beta = np.random.randn(xdim, ydim)

        Rho = boom.random_correlation_matrix(ydim).to_numpy()
        S = np.diag(R.rgamma(ydim, 1, 1))
        self._Sigma = S @ Rho @ S
        Sigma_root = np.linalg.cholesky(self._Sigma)

        errors = (Sigma_root @ np.random.randn(ydim, sample_size)).T
        encoder = boom.DatasetEncoder(encoders)
        xcat = encoder.encode_dataset(R.to_boom_data_table(
            pd.DataFrame(cats)))
        yhat = xcat.to_numpy() @ self._beta
        numerics = yhat + errors
        self._data = pd.DataFrame(
            numerics,
            columns=["X" + str(i + 1) for i in range(ydim)]
        )
        for vname, column in cats.items():
            self._data[vname] = column
        self._ydim = ydim
        self._xdim = xdim
        self._ncat = num_cat

    def modify_dataset(self, nclusters):
        pass

    def test_data_table(self):
        table = R.to_boom_data_table(self._data)
        self.assertEqual(table.nrow, self._data.shape[0])
        self.assertEqual(table.ncol, self._data.shape[1])

        frame = R.to_pd_dataframe(table)
        for i in range(5):
            self.assertTrue(np.all(self._data.iloc[:, i] == frame.iloc[:, i]))

    def test_datetime_support(self):
        table = self._data
        epoch = pd.Timestamp("2024-05-15", )
        dates = pd.date_range("2024-05-15", periods = table.shape[0], freq="D")
        table["dates"] = dates

        # Check that the round trip with boom preseves the dates unaltered.
        boom_dates = R.to_boom_datetime_vector(pd.Series(dates))
        py_dates = R.to_pd_datetime64(boom_dates)
        self.assertTrue(np.all(py_dates == dates))
        

        boom_table = R.to_boom_data_table(table)
        frame = R.to_pd_dataframe(boom_table)
        self.assertTrue(np.allclose(frame.shape, table.shape))
        self.assertTrue(np.all(frame.columns == table.columns))
        for colname in table.columns:
            self.assertTrue(np.all(table[colname] == frame[colname]))


_debug_mode = False

if _debug_mode:
    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = DataTableTest()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_datetime_support()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

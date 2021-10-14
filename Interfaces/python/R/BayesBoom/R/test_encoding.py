import unittest
import BayesBoom.R as R
import numpy as np
import pandas as pd


class TestEffectEncoder(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def test_init(self):
        encoder = R.EffectEncoder("Color", ["Red", "Blue", "Green"])
        self.assertEqual(encoder._levels, ["Red", "Blue"])
        self.assertEqual(encoder._baseline, "Green")
        self.assertEqual(2, encoder.dim)

        self.assertEqual(["Color.Red", "Color.Blue"],
                         encoder.encoded_variable_names)

        encoder_rep = str(encoder)
        self.assertEqual("EffectEncoder for Color", encoder_rep)

    def test_encoding(self):
        encoder = R.EffectEncoder("Color", ["Red", "Blue", "Green"])
        x = ["Red", "Blue", "Green"]
        enc = encoder.encode(x)
        expected = np.array([[1.0, 0.0],
                             [0.0, 1.0],
                             [-1.0, -1.0]])
        self.assertTrue(np.allclose(enc, expected))

    def test_new_level(self):
        encoder = R.EffectEncoder("Color", ["Red", "Blue", "Green"])
        enc = encoder.encode(["Red", "Blue", "Orange", np.NaN, 17, "Green"])
        expected = np.array([[1.0, 0.0],
                             [0.0, 1.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [-1.0, -1.0]])
        self.assertTrue(np.allclose(enc, expected))

    def test_encode_dataset(self):
        data = pd.DataFrame(np.random.randn(3, 2),
                            columns=["X1", "X2"])
        data["Color"] = ["Red", "Blue", "Green"]
        encoder = R.EffectEncoder("Color", ["Red", "Blue", "Green"])
        enc = encoder.encode_dataset(data)
        expected = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0]])
        self.assertTrue(np.allclose(enc, expected))


class TestDatasetEncoder(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_init(self):
        enc1 = R.EffectEncoder("Color", ["Red", "Blue"])
        enc2 = R.IdentityEncoder("Height")
        enc3 = R.InteractionEncoder(enc1, enc2)
        d = R.DatasetEncoder([enc1, enc2, enc3])
        self.assertIsInstance(d, R.DatasetEncoder)

    def test_encoding(self):
        enc1 = R.EffectEncoder("Color", ["Red", "Blue"])
        enc2 = R.IdentityEncoder("Height")
        enc3 = R.InteractionEncoder(enc1, enc2)
        encoder = R.DatasetEncoder([enc1, enc2, enc3])

        sample_size = 1000
        data = pd.DataFrame(
            {
                "Height": np.random.randn(sample_size),
                "Color": np.random.choice(["Red", "Blue"], sample_size)
            }
        )
        enc = encoder.encode_dataset(data)
        self.assertEqual(sample_size, enc.shape[0])
        self.assertEqual(4, enc.shape[1])
        self.assertTrue(np.allclose(enc[:, 2], data.iloc[:, 0]))


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

    rig = TestDatasetEncoder()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_init()
    rig.test_encoding()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

import unittest
import BayesBoom.R as R
import numpy as np
import pandas as pd
import json


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

    def test_json(self):
        encoder = R.EffectEncoder("Color", ["Red", "Blue", "Green"])
        data = ["Red", "Blue", "Orange", np.NaN, 17, "Green"]
        enc = encoder.encode(data)
#         json_encoder = R.MainEffectEncoderJsonEncoder()
        json_string = json.dumps(encoder,
                                 cls=R.MainEffectEncoderJsonEncoder)
        encoder2 = json.loads(json_string, cls=R.MainEffectEncoderJsonDecoder)
        enc2 = encoder2.encode(data)
        self.assertTrue(np.allclose(enc, enc2))


class TestOneHotEncoder(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_encoding(self):
        levels = ["Red", "Blue", "Green"]
        encoder = R.OneHotEncoder("blah", levels, baseline_level="Green")
        self.assertEqual(2, encoder.dim)
        x = ["Red", "Red", "Green", "Blue"]
        X = encoder.encode(x)
        self.assertEqual((4, 2), X.shape)
        expected = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0]
        ])
        self.assertTrue(np.allclose(X, expected))

    def test_json(self):
        levels = ["Red", "Blue", "Green"]
        encoder = R.OneHotEncoder("blah", levels, baseline_level="Green")
        data = ["Red", "Red", "Green", "Blue"]
        output = encoder.encode(data)

        json_string = json.dumps(encoder, cls=R.MainEffectEncoderJsonEncoder)
        encoder2 = json.loads(json_string, cls=R.MainEffectEncoderJsonDecoder)
        self.assertTrue(np.allclose(output, encoder2.encode(data)))


class TestIdentityEncoder(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_everything(self):
        encoder = R.IdentityEncoder("Blah")
        self.assertEqual(encoder.variable_name, "Blah")
        x = np.random.randn(4)
        output = encoder.encode(x)
        self.assertEqual(output.shape, (4, 1))
        self.assertTrue(np.allclose(output.ravel(), x))

        json_string = json.dumps(encoder, cls=R.MainEffectEncoderJsonEncoder)
        encoder2 = json.loads(json_string, cls=R.MainEffectEncoderJsonDecoder)
        self.assertTrue(np.allclose(output, encoder2.encode(x)))


class TestSuccessEncoder(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_encoding(self):
        encoder = R.SuccessEncoder("Blah", ["A", "C"])
        data = ["A", "B", "C", "D"]
        enc = encoder.encode(data)
        self.assertEqual(enc.shape, (4, 1))
        self.assertTrue(np.allclose(
            enc.ravel(), np.array([1, 0, 1, 0])))

    def test_json(self):
        encoder = R.SuccessEncoder("Blah", ["A", "C"])
        data = ["A", "B", "C", "D"]
        enc = encoder.encode(data)

        json_string = json.dumps(encoder, cls=R.MainEffectEncoderJsonEncoder)
        encoder2 = json.loads(json_string, cls=R.MainEffectEncoderJsonDecoder)
        self.assertTrue(np.allclose(enc, encoder2.encode(data)))


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

        json_string = json.dumps(encoder, cls=R.DatasetEncoderJsonEncoder)
        encoder2 = json.loads(json_string, cls=R.DatasetEncoderJsonDecoder)
        self.assertTrue(np.allclose(enc, encoder2.encode_dataset(data)))


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

    rig.test_encoding()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

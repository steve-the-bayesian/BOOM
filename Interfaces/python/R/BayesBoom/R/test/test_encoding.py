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
        enc = encoder.encode(["Red", "Blue", "Orange", np.nan, 17, "Green"])
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

    def test_boom_equivalence(self):
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

        boom_enc = encoder.boom()
        boom_data = R.to_boom_data_table(data)
        boom_predictors = boom_enc.encode_dataset(boom_data)
        self.assertTrue(np.allclose(boom_predictors.to_numpy(),
                                    expected))

    def test_json(self):
        encoder = R.EffectEncoder("Color", ["Red", "Blue", "Green"])
        data = ["Red", "Blue", "Orange", np.nan, 17, "Green"]
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


class TestTransformationEncoder(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)
        self.x = np.array([1.0, 2.0, 4.0, 9.0])

    def test_identity(self):
        enc = R.TransformationEncoder("price", "identity")
        self.assertEqual(enc.variable_name, "price")
        self.assertEqual(enc.transform_name, "identity")
        self.assertEqual(enc.dim, 1)
        self.assertEqual(enc.encoded_variable_names, ["identity(price)"])

        out = enc.encode(self.x)
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(np.allclose(out.ravel(), self.x))

    def test_log1p(self):
        enc = R.TransformationEncoder("price", "log1p")
        self.assertEqual(enc.encoded_variable_names, ["log1p(price)"])
        out = enc.encode(self.x)
        self.assertTrue(np.allclose(out.ravel(), np.log1p(self.x)))

    def test_sqrt(self):
        enc = R.TransformationEncoder("price", "sqrt")
        out = enc.encode(self.x)
        self.assertTrue(np.allclose(out.ravel(), np.sqrt(self.x)))

    def test_log(self):
        enc = R.TransformationEncoder("price", "log")
        out = enc.encode(self.x)
        self.assertTrue(np.allclose(out.ravel(), np.log(self.x)))

    def test_all_builtin_names(self):
        builtins = ["identity", "log", "log1p", "log2", "log10",
                    "sqrt", "exp", "expm1", "abs", "square"]
        for name in builtins:
            enc = R.TransformationEncoder("x", name)
            self.assertEqual(enc.transform_name, name)

    def test_user_defined_functor(self):
        cube = lambda x: x ** 3
        enc = R.TransformationEncoder("x", ("cube", cube))
        self.assertEqual(enc.transform_name, "cube")
        out = enc.encode(self.x)
        self.assertTrue(np.allclose(out.ravel(), self.x ** 3))

    def test_register_transformation(self):
        R.register_transformation("recip", lambda x: 1.0 / x)
        enc = R.TransformationEncoder("x", "recip")
        out = enc.encode(self.x)
        self.assertTrue(np.allclose(out.ravel(), 1.0 / self.x))

    def test_unknown_transform_raises(self):
        with self.assertRaises(ValueError):
            R.TransformationEncoder("x", "no_such_transform")

    def test_json_roundtrip_builtin(self):
        enc = R.TransformationEncoder("price", "log1p")
        json_string = json.dumps(enc, cls=R.MainEffectEncoderJsonEncoder)
        enc2 = json.loads(json_string, cls=R.MainEffectEncoderJsonDecoder)
        self.assertIsInstance(enc2, R.TransformationEncoder)
        self.assertEqual(enc2.variable_name, "price")
        self.assertEqual(enc2.transform_name, "log1p")
        self.assertTrue(np.allclose(enc2.encode(self.x), enc.encode(self.x)))

    def test_json_roundtrip_user_defined(self):
        # User-defined transform: must be in TRANSFORMATION_ENCODER_REGISTRY
        # before the decoder runs (simulated here by registering first).
        R.register_transformation("negate", lambda x: -x)
        enc = R.TransformationEncoder("x", "negate")

        json_string = json.dumps(enc, cls=R.MainEffectEncoderJsonEncoder)
        enc2 = json.loads(json_string, cls=R.MainEffectEncoderJsonDecoder)
        self.assertEqual(enc2.transform_name, "negate")
        self.assertTrue(np.allclose(enc2.encode(self.x), -self.x))

    def test_json_unknown_transform_raises_on_decode(self):
        # Manually craft a JSON payload for a transform not in the registry.
        payload = json.dumps({
            "type": "TransformationEncoder",
            "encoder": {
                "variable_name": "x",
                "transform": "__unregistered__",
            },
        })
        with self.assertRaises(ValueError):
            json.loads(payload, cls=R.MainEffectEncoderJsonDecoder)


class TestMissingDummyEncoder(unittest.TestCase):
    def setUp(self):
        np.random.seed(8675309)

    def test_numeric(self):
        base = R.IdentityEncoder("Blah")
        encoder = R.MissingDummyEncoder(base)
        self.assertEqual(encoder.variable_name, "Blah")

        x = np.array([1, 2, np.nan, 3])
        foo = encoder.encode(x)
        self.assertEqual(foo.shape, (4, 2))
        self.assertTrue(np.allclose(foo[:, 0], np.array([0, 0, 1, 0])))
        self.assertTrue(np.allclose(foo[:, 1], np.array([1, 2, 0, 3])))

        y = np.array([1, 2, 3])
        bar = encoder.encode(y)
        self.assertEqual(bar.shape, (3, 2))
        self.assertTrue(np.allclose(bar[:, 0], np.array([0, 0, 0])))
        self.assertTrue(np.allclose(bar[:, 1], np.array([1, 2, 3])))

    def test_categorical(self):
        base = R.EffectEncoder("Stooges", ["Larry", "Moe", "Curly"])
        encoder = R.MissingDummyEncoder(base)
        x = np.array(["Larry", "Curly", np.nan, "Moe"], dtype=object)
        foo = encoder.encode(x)
        self.assertEqual(foo.shape, (4, 3))
        self.assertTrue(np.allclose(
            foo,
            np.array([[0, 1, 0],
                      [0, -1, -1],
                      [1, 0, 0],
                      [0, 0, 1]])))


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

    def test_simulation(self):
        sample_size = 100
        colors = ["Red", "Green", "Blue"]
        color_encoder = R.EffectEncoder("Color", colors)
        shapes = ["Circle", "Triangle", "Pentagon", "Octogon"]
        shape_encoder = R.OneHotEncoder("Shape", shapes, baseline_level="Circle")

        height_encoder = R.IdentityEncoder("Height")

        enc = R.DatasetEncoder([color_encoder, shape_encoder, height_encoder])
        data = enc.simulate(sample_size)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[0], sample_size)
        self.assertEqual(data.shape[1], 3)
        self.assertEqual(data.columns[0], "Color")
        self.assertEqual(data.columns[1], "Shape")
        self.assertEqual(data.columns[2], "Height")

        color_table = R.table(data["Color"])
        self.assertEqual(len(color_table), len(colors))

        shape_table = R.table(data["Shape"])
        self.assertEqual(len(shape_table), len(shapes))

        self.assertEqual(data["Height"].dtype, "float64")

    def test_boom_encoder_creation(self):
        sample_size = 100
        colors = ["Red", "Green", "Blue"]
        color_encoder = R.EffectEncoder("Color", colors)
        shapes = ["Circle", "Triangle", "Pentagon", "Octogon"]
        # shape_encoder = R.OneHotEncoder("Shape", shapes, baseline_level="Circle")
        shape_encoder = R.EffectEncoder("Shape", shapes, baseline_level="Circle")

        height_encoder = R.IdentityEncoder("Height")

        enc = R.DatasetEncoder([color_encoder, shape_encoder, height_encoder])
        data = enc.simulate(sample_size)
        predictors = enc.encode_dataset(data)

        boom_data = R.to_boom_data_table(data)
        boom_enc = enc.boom()
        boom_predictors = boom_enc.encode_dataset(boom_data)

        boom_names = boom_enc.encoded_variable_names
        py_names = [x.replace(".", ":") for x in enc.encoded_variable_names]
        for i in range(len(boom_names)):
            self.assertEqual(py_names[i], boom_names[i])

        print(boom_enc.encoded_variable_names)
        print(enc.encoded_variable_names)

        self.assertEqual(data.shape[0], boom_data.nrow)
        self.assertEqual(data.shape[1], boom_data.ncol)
        self.assertTrue(np.allclose(boom_predictors.to_numpy(), predictors))


_debug_mode = False

if _debug_mode:
    import pdb  # noqa

    # Turn warnings into errors.
    # warnings.simplefilter("error")

    # Run the test you are trying to debug here.  Instantiate the test class,
    # then call the problematic test.  Call pdb.pm() in the event of an
    # exception.
    print("Hello, world!")

#    rig = TestDatasetEncoder()
    rig = TestEffectEncoder()
    if hasattr(rig, "setUpClass"):
        rig.setUpClass()
    if hasattr(rig, "setUp"):
        rig.setUp()

    rig.test_boom_equivalence()

    print("Goodbye, cruel world!")

else:
    if __name__ == "__main__":
        unittest.main(verbosity=2)

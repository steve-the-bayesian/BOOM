import unittest
import numpy as np
import pandas as pd

import BayesBoom.R as R
from BayesBoom.bandits.linear_bandit_encoder import (
    ExperimentStructure,
    ArmMap,
    ExperimentArmEncoder,
)


def _make_color_experiment():
    xp = ExperimentStructure()
    xp.add_factor("Color", ["Red", "Green", "Blue"])
    return xp


def _make_two_factor_experiment():
    xp = ExperimentStructure()
    xp.add_factor("InterestRate", ["0.0", "0.1", "0.2", "0.3"])
    xp.add_factor("CreditLimit", ["0", "10", "20", "50", "100"])
    return xp


class TestExperimentArmEncoderDim(unittest.TestCase):
    """dim is number_of_levels - 1 (effects coding drops one level)."""

    def test_dim_three_levels(self):
        xp = _make_color_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("Color", arm_map, "Red")
        # 3 levels - 1 baseline = 2
        self.assertEqual(2, enc.dim)

    def test_dim_four_levels(self):
        xp = _make_two_factor_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("InterestRate", arm_map, "0.0")
        # 4 levels - 1 = 3
        self.assertEqual(3, enc.dim)

    def test_python_dim_matches_boom_dim(self):
        xp = _make_two_factor_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("InterestRate", arm_map, "0.0")
        self.assertEqual(enc.dim, enc.boom().dim)

    def test_python_dim_matches_boom_dim_credit_limit(self):
        xp = _make_two_factor_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("CreditLimit", arm_map, "0")
        # 5 levels - 1 = 4
        self.assertEqual(4, enc.dim)
        self.assertEqual(enc.dim, enc.boom().dim)


class TestExperimentArmEncoderEncodeDataset(unittest.TestCase):
    """Python encode_dataset and boom encode_dataset must agree."""

    def _encode_both(self, enc, data):
        py_result = enc.encode_dataset(data)
        bdt = R.to_boom_data_table(data)
        boom_result = R.to_numpy(enc.boom().encode_dataset(bdt))
        return py_result, boom_result

    def test_baseline_row_is_all_minus_one(self):
        xp = _make_color_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("Color", arm_map, "Red")
        data = pd.DataFrame({"Color": ["Red"]})
        py, bm = self._encode_both(enc, data)
        np.testing.assert_array_equal(np.array([[-1.0, -1.0]]), py)
        np.testing.assert_array_equal(py, bm)

    def test_first_nonbaseline_row(self):
        xp = _make_color_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("Color", arm_map, "Red")
        data = pd.DataFrame({"Color": ["Green"]})
        py, bm = self._encode_both(enc, data)
        np.testing.assert_array_equal(np.array([[1.0, 0.0]]), py)
        np.testing.assert_array_equal(py, bm)

    def test_second_nonbaseline_row(self):
        xp = _make_color_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("Color", arm_map, "Red")
        data = pd.DataFrame({"Color": ["Blue"]})
        py, bm = self._encode_both(enc, data)
        np.testing.assert_array_equal(np.array([[0.0, 1.0]]), py)
        np.testing.assert_array_equal(py, bm)

    def test_multiple_rows_python_equals_boom(self):
        xp = _make_color_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("Color", arm_map, "Red")
        data = pd.DataFrame({"Color": ["Red", "Green", "Blue", "Green", "Red"]})
        py, bm = self._encode_both(enc, data)
        np.testing.assert_array_almost_equal(py, bm)

    def test_all_rows_python_equals_boom(self):
        xp = _make_color_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("Color", arm_map, "Red")
        data = pd.DataFrame({"Color": ["Red", "Green", "Blue"]})
        py, bm = self._encode_both(enc, data)
        expected = np.array([[-1., -1.], [1., 0.], [0., 1.]])
        np.testing.assert_array_almost_equal(expected, py)
        np.testing.assert_array_almost_equal(py, bm)

    def test_four_level_factor_python_equals_boom(self):
        xp = _make_two_factor_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("InterestRate", arm_map, "0.0")
        data = pd.DataFrame({
            "InterestRate": ["0.0", "0.1", "0.2", "0.3", "0.0"],
            "CreditLimit": ["0", "10", "20", "50", "100"],
        })
        py, bm = self._encode_both(enc, data)
        np.testing.assert_array_almost_equal(py, bm)

    def test_baseline_row_all_minus_one_four_levels(self):
        xp = _make_two_factor_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("InterestRate", arm_map, "0.0")
        data = pd.DataFrame({"InterestRate": ["0.0"], "CreditLimit": ["0"]})
        py, bm = self._encode_both(enc, data)
        np.testing.assert_array_equal(np.array([[-1., -1., -1.]]), py)
        np.testing.assert_array_equal(py, bm)

    def test_output_shape(self):
        xp = _make_two_factor_experiment()
        arm_map = ArmMap(xp)
        enc = ExperimentArmEncoder("InterestRate", arm_map, "0.0")
        data = pd.DataFrame({
            "InterestRate": ["0.0", "0.1", "0.2"],
            "CreditLimit": ["0", "10", "20"],
        })
        py = enc.encode_dataset(data)
        self.assertEqual((3, enc.dim), py.shape)


class TestExperimentArmEncoderMetadata(unittest.TestCase):

    def test_required_variables(self):
        xp = _make_color_experiment()
        enc = ExperimentArmEncoder("Color", ArmMap(xp), "Red")
        self.assertEqual(["Color"], enc.required_variables)

    def test_encodes_own_variable(self):
        xp = _make_color_experiment()
        enc = ExperimentArmEncoder("Color", ArmMap(xp), "Red")
        self.assertTrue(enc.encodes("Color"))

    def test_encodes_other_variable_false(self):
        xp = _make_color_experiment()
        enc = ExperimentArmEncoder("Color", ArmMap(xp), "Red")
        self.assertFalse(enc.encodes("Size"))

    def test_extract_main_effects(self):
        xp = _make_color_experiment()
        enc = ExperimentArmEncoder("Color", ArmMap(xp), "Red")
        me = enc.extract_main_effects()
        self.assertIn("Color", me)
        self.assertIs(enc, me["Color"])

    def test_encoded_variable_names_count(self):
        xp = _make_color_experiment()
        enc = ExperimentArmEncoder("Color", ArmMap(xp), "Red")
        self.assertEqual(enc.dim, len(enc.encoded_variable_names))

    def test_encoded_variable_names_prefix(self):
        xp = _make_color_experiment()
        enc = ExperimentArmEncoder("Color", ArmMap(xp), "Red")
        for name in enc.encoded_variable_names:
            self.assertTrue(name.startswith("Color["))

    def test_boom_encoded_variable_names_count(self):
        xp = _make_color_experiment()
        enc = ExperimentArmEncoder("Color", ArmMap(xp), "Red")
        self.assertEqual(enc.dim, len(enc.boom().encoded_variable_names))


if __name__ == "__main__":
    unittest.main()

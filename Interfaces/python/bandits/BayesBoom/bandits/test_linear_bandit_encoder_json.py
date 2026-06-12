import json
import unittest
import numpy as np

import BayesBoom.R as R
from BayesBoom.bandits.linear_bandit_encoder import (
    ExperimentStructure,
    ExperimentStructureJSONEncoder,
    ExperimentStructureJSONDecoder,
    ArmMap,
    ArmMapJsonEncoder,
    ArmMapJsonDecoder,
    ExperimentArmEncoder,
    LinearBanditEncoder,
    LinearBanditEncoderJSONEncoder,
    LinearBanditEncoderJSONDecoder,
)


def _two_factor_experiment():
    """ButtonPosition x ButtonColor, 2x2 = 4 arms."""
    xp = ExperimentStructure()
    xp.add_factor("ButtonPosition", ["Left", "Right"])
    xp.add_factor("ButtonColor", ["Red", "Blue"])
    return xp


def _three_level_experiment():
    """Color x Size, 3x2 = 6 arms."""
    xp = ExperimentStructure()
    xp.add_factor("Color", ["Red", "Green", "Blue"])
    xp.add_factor("Size", ["Small", "Large"])
    return xp


def _round_trip_experiment_structure(xp):
    payload = ExperimentStructureJSONEncoder().default(xp)
    json_string = json.dumps(payload)
    return ExperimentStructureJSONDecoder().decode(json_string)


def _round_trip_arm_map(arm_map):
    payload = ArmMapJsonEncoder().default(arm_map)
    json_string = json.dumps(payload)
    return ArmMapJsonDecoder().decode(json_string)


def _round_trip_linear_bandit_encoder(encoder):
    payload = LinearBanditEncoderJSONEncoder().default(encoder)
    json_string = json.dumps(payload)
    return LinearBanditEncoderJSONDecoder().decode(json_string)


# ===========================================================================
class TestExperimentStructureJSON(unittest.TestCase):

    def test_single_factor_names(self):
        xp = ExperimentStructure()
        xp.add_factor("Color", ["Red", "Green", "Blue"])
        restored = _round_trip_experiment_structure(xp)
        self.assertEqual(["Color"], restored._factor_names)

    def test_single_factor_levels(self):
        xp = ExperimentStructure()
        xp.add_factor("Color", ["Red", "Green", "Blue"])
        restored = _round_trip_experiment_structure(xp)
        self.assertEqual([["Red", "Green", "Blue"]], restored._factor_levels)

    def test_multiple_factor_names(self):
        xp = _two_factor_experiment()
        restored = _round_trip_experiment_structure(xp)
        self.assertEqual(["ButtonPosition", "ButtonColor"],
                         restored._factor_names)

    def test_multiple_factor_levels(self):
        xp = _two_factor_experiment()
        restored = _round_trip_experiment_structure(xp)
        self.assertEqual([["Left", "Right"], ["Red", "Blue"]],
                         restored._factor_levels)

    def test_empty_structure(self):
        xp = ExperimentStructure()
        restored = _round_trip_experiment_structure(xp)
        self.assertEqual([], restored._factor_names)
        self.assertEqual([], restored._factor_levels)

    def test_three_factor_round_trip(self):
        xp = ExperimentStructure()
        xp.add_factor("A", ["a1", "a2"])
        xp.add_factor("B", ["b1", "b2", "b3"])
        xp.add_factor("C", ["c1", "c2", "c3", "c4"])
        restored = _round_trip_experiment_structure(xp)
        self.assertEqual(xp._factor_names, restored._factor_names)
        self.assertEqual(xp._factor_levels, restored._factor_levels)


# ===========================================================================
class TestArmMapJSON(unittest.TestCase):

    def test_number_of_arms_preserved(self):
        arm_map = ArmMap(_two_factor_experiment())
        restored = _round_trip_arm_map(arm_map)
        self.assertEqual(arm_map.number_of_arms, restored.number_of_arms)

    def test_six_arm_experiment(self):
        arm_map = ArmMap(_three_level_experiment())
        restored = _round_trip_arm_map(arm_map)
        self.assertEqual(6, restored.number_of_arms)

    def test_factor_level_matrix_shape(self):
        arm_map = ArmMap(_two_factor_experiment())
        restored = _round_trip_arm_map(arm_map)
        self.assertEqual(arm_map.map.shape, restored.map.shape)

    def test_factor_level_matrix_values(self):
        arm_map = ArmMap(_two_factor_experiment())
        restored = _round_trip_arm_map(arm_map)
        np.testing.assert_array_equal(arm_map.map, restored.map)

    def test_factor_names_in_experiment_structure(self):
        arm_map = ArmMap(_three_level_experiment())
        restored = _round_trip_arm_map(arm_map)
        self.assertEqual(
            arm_map._experiment_structure._factor_names,
            restored._experiment_structure._factor_names)


# ===========================================================================
class TestLinearBanditEncoderJSONNoContext(unittest.TestCase):
    """LinearBanditEncoder with only ExperimentArmEncoders (no context)."""

    def _make_encoder(self):
        xp = _two_factor_experiment()
        arm_map = ArmMap(xp)
        pos_enc = ExperimentArmEncoder("ButtonPosition", arm_map)
        col_enc = ExperimentArmEncoder("ButtonColor", arm_map)
        dataset_enc = R.DatasetEncoder([pos_enc, col_enc])
        return LinearBanditEncoder(arm_map, dataset_enc)

    def setUp(self):
        self.encoder = self._make_encoder()
        self.restored = _round_trip_linear_bandit_encoder(self.encoder)

    def test_number_of_arms(self):
        self.assertEqual(self.encoder.number_of_arms,
                         self.restored.number_of_arms)

    def test_dim(self):
        self.assertEqual(self.encoder.dim, self.restored.dim)

    def test_encode_row_all_arms(self):
        for arm in range(self.encoder.number_of_arms):
            original_row = self.encoder.encode_row(arm, None)
            restored_row = self.restored.encode_row(arm, None)
            np.testing.assert_array_almost_equal(
                original_row, restored_row,
                err_msg=f"encode_row mismatch for arm {arm}")

    def test_encode_row_length(self):
        row = self.restored.encode_row(0, None)
        self.assertEqual(self.restored.dim, len(row))

    def test_arm_rows_are_distinct(self):
        rows = [self.restored.encode_row(a, None)
                for a in range(self.restored.number_of_arms)]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                self.assertFalse(
                    np.allclose(rows[i], rows[j]),
                    msg=f"Arms {i} and {j} have identical encodings after round-trip")


# ===========================================================================
class TestLinearBanditEncoderJSONWithIdentityContext(unittest.TestCase):
    """LinearBanditEncoder with ExperimentArmEncoders + IdentityEncoder."""

    def _make_encoder(self):
        xp = _two_factor_experiment()
        arm_map = ArmMap(xp)
        pos_enc = ExperimentArmEncoder("ButtonPosition", arm_map)
        col_enc = ExperimentArmEncoder("ButtonColor", arm_map)
        x1_enc = R.IdentityEncoder("x1")
        dataset_enc = R.DatasetEncoder([pos_enc, col_enc, x1_enc])
        return LinearBanditEncoder(arm_map, dataset_enc)

    def setUp(self):
        import pandas as pd
        self.encoder = self._make_encoder()
        self.restored = _round_trip_linear_bandit_encoder(self.encoder)
        self.ctx = pd.DataFrame({"x1": [2.5]})

    def test_number_of_arms(self):
        self.assertEqual(self.encoder.number_of_arms,
                         self.restored.number_of_arms)

    def test_dim(self):
        # intercept(1) + ButtonPosition(1) + ButtonColor(1) + x1(1) = 4
        self.assertEqual(4, self.restored.dim)

    def test_encode_row_all_arms(self):
        for arm in range(self.encoder.number_of_arms):
            original_row = self.encoder.encode_row(arm, self.ctx)
            restored_row = self.restored.encode_row(arm, self.ctx)
            np.testing.assert_array_almost_equal(
                original_row, restored_row,
                err_msg=f"encode_row mismatch for arm {arm}")

    def test_context_column_preserved(self):
        row = self.restored.encode_row(0, self.ctx)
        # Last column should be the x1 value (2.5)
        self.assertAlmostEqual(2.5, row[-1])


# ===========================================================================
class TestLinearBanditEncoderJSONWithEffectEncoder(unittest.TestCase):
    """LinearBanditEncoder with ExperimentArmEncoders + EffectEncoder.

    EffectEncoder context variables are not compatible with encode_row (boom
    expects categorical data, not plain strings, in MixedMultivariateData).
    These tests only verify that JSON round-trip preserves the structural
    properties of the encoder.
    """

    def _make_encoder(self):
        xp = _two_factor_experiment()
        arm_map = ArmMap(xp)
        pos_enc = ExperimentArmEncoder("ButtonPosition", arm_map)
        col_enc = ExperimentArmEncoder("ButtonColor", arm_map)
        device_enc = R.EffectEncoder("Device", ["Mobile", "Tablet", "Desktop"])
        dataset_enc = R.DatasetEncoder([pos_enc, col_enc, device_enc])
        return LinearBanditEncoder(arm_map, dataset_enc)

    def setUp(self):
        self.encoder = self._make_encoder()
        self.restored = _round_trip_linear_bandit_encoder(self.encoder)

    def test_number_of_arms(self):
        self.assertEqual(self.encoder.number_of_arms,
                         self.restored.number_of_arms)

    def test_dim(self):
        # intercept(1) + ButtonPosition(1) + ButtonColor(1) + Device(2) = 5
        self.assertEqual(5, self.restored.dim)

    def test_json_is_valid_string(self):
        payload = LinearBanditEncoderJSONEncoder().default(self.encoder)
        json_string = json.dumps(payload)
        parsed = json.loads(json_string)
        self.assertIn("arm_map", parsed)
        self.assertIn("dataset_encoder", parsed)

    def test_encoder_types_in_payload(self):
        payload = LinearBanditEncoderJSONEncoder().default(self.encoder)
        enc_types = payload["dataset_encoder"]["encoder_types"]
        self.assertIn("ExperimentArmEncoder", enc_types)
        self.assertIn("EffectEncoder", enc_types)


# ===========================================================================
class TestLinearBanditEncoderJSONMixedContext(unittest.TestCase):
    """LinearBanditEncoder with a 3-level experiment and two numeric context
    variables, testing JSON round-trip including encode_row."""

    def _make_encoder(self):
        xp = _three_level_experiment()
        arm_map = ArmMap(xp)
        color_enc = ExperimentArmEncoder("Color", arm_map)
        size_enc = ExperimentArmEncoder("Size", arm_map)
        score_enc = R.IdentityEncoder("score")
        temp_enc = R.IdentityEncoder("temperature")
        dataset_enc = R.DatasetEncoder([color_enc, size_enc, score_enc,
                                        temp_enc])
        return LinearBanditEncoder(arm_map, dataset_enc)

    def setUp(self):
        import pandas as pd
        self.encoder = self._make_encoder()
        self.restored = _round_trip_linear_bandit_encoder(self.encoder)
        self.ctx = pd.DataFrame({"score": [0.7], "temperature": [22.0]})

    def test_number_of_arms(self):
        self.assertEqual(6, self.restored.number_of_arms)

    def test_dim(self):
        # intercept(1) + Color(2) + Size(1) + score(1) + temperature(1) = 6
        # Color has 3 levels so effects coding produces 3-1=2 columns.
        self.assertEqual(6, self.restored.dim)

    def test_encode_row_all_arms(self):
        for arm in range(self.encoder.number_of_arms):
            original_row = self.encoder.encode_row(arm, self.ctx)
            restored_row = self.restored.encode_row(arm, self.ctx)
            np.testing.assert_array_almost_equal(
                original_row, restored_row,
                err_msg=f"encode_row mismatch for arm {arm}")

    def test_encoder_types_in_payload(self):
        payload = LinearBanditEncoderJSONEncoder().default(self.encoder)
        enc_types = payload["dataset_encoder"]["encoder_types"]
        self.assertIn("ExperimentArmEncoder", enc_types)
        self.assertIn("IdentityEncoder", enc_types)

    def test_json_is_valid_string(self):
        payload = LinearBanditEncoderJSONEncoder().default(self.encoder)
        json_string = json.dumps(payload)
        parsed = json.loads(json_string)
        self.assertIn("arm_map", parsed)
        self.assertIn("dataset_encoder", parsed)


if __name__ == "__main__":
    unittest.main()

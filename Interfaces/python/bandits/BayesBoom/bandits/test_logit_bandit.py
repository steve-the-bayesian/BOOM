import unittest
import numpy as np
import pandas as pd

import BayesBoom.R as R
from BayesBoom.bandits.linear_bandit_encoder import (
    ExperimentStructure,
    ArmMap,
    ExperimentArmEncoder,
    LinearBanditEncoder,
)
from BayesBoom.bandits.logit_bandit import LogitBandit


def _make_bandit_no_context():
    """2-factor experiment: ButtonPosition x ButtonColor, no context."""
    xp = ExperimentStructure()
    xp.add_factor("ButtonPosition", ["Left", "Right"])
    xp.add_factor("ButtonColor", ["Red", "Blue"])
    arm_map = ArmMap(xp)
    pos_enc = ExperimentArmEncoder("ButtonPosition", arm_map)
    col_enc = ExperimentArmEncoder("ButtonColor", arm_map)
    dataset_encoder = R.DatasetEncoder([pos_enc, col_enc])
    encoder = LinearBanditEncoder(arm_map, dataset_encoder)
    return LogitBandit(arm_map, encoder)


def _make_bandit_with_context():
    """2-factor experiment: ButtonPosition x ButtonColor, plus numeric x1."""
    xp = ExperimentStructure()
    xp.add_factor("ButtonPosition", ["Left", "Right"])
    xp.add_factor("ButtonColor", ["Red", "Blue"])
    arm_map = ArmMap(xp)
    pos_enc = ExperimentArmEncoder("ButtonPosition", arm_map)
    col_enc = ExperimentArmEncoder("ButtonColor", arm_map)
    x1_enc = R.IdentityEncoder("x1")
    dataset_encoder = R.DatasetEncoder([pos_enc, col_enc, x1_enc])
    encoder = LinearBanditEncoder(arm_map, dataset_encoder)
    return LogitBandit(arm_map, encoder)


class TestLogitBanditNoContext(unittest.TestCase):

    def setUp(self):
        self.bandit = _make_bandit_no_context()

    def test_number_of_arms(self):
        # 2 positions x 2 colors = 4 arms
        self.assertEqual(4, self.bandit.number_of_arms)

    def test_observe_data_accumulates(self):
        self.bandit.observe_data(0, 3, 5)
        self.bandit.observe_data(1, 1, 5)
        self.bandit.observe_data(2, 2, 5)
        self.assertEqual(3, len(self.bandit._training_data))

    def test_observe_data_rejects_non_dataframe_context(self):
        with self.assertRaises(TypeError):
            self.bandit.observe_data(0, 1, 2, context={"x": 1})

    def test_ndraws_before_update(self):
        self.bandit.observe_data(0, 5, 10)
        self.assertEqual(0, self.bandit.ndraws)

    def test_update_posterior(self):
        self.bandit.observe_data(0, 5, 10)
        self.bandit.observe_data(1, 2, 10)
        self.bandit.update_posterior(200)
        self.assertEqual(200, self.bandit.ndraws)

    def test_optimal_arm_probabilities_shape_and_sum(self):
        self.bandit.observe_data(0, 90, 100)
        self.bandit.observe_data(1, 30, 100)
        self.bandit.observe_data(2, 20, 100)
        self.bandit.observe_data(3, 10, 100)
        self.bandit.update_posterior(1000)
        probs = self.bandit.optimal_arm_probabilities()
        self.assertEqual(4, len(probs))
        self.assertAlmostEqual(1.0, probs.sum(), places=10)

    def test_optimal_arm_probabilities_arm0_dominates(self):
        # Arm 0 (Left, Red) gets far more successes; it should dominate.
        self.bandit.observe_data(0, 90, 100)
        self.bandit.observe_data(1, 30, 100)
        self.bandit.observe_data(2, 20, 100)
        self.bandit.observe_data(3, 10, 100)
        self.bandit.update_posterior(1000)
        probs = self.bandit.optimal_arm_probabilities()
        for i in range(1, 4):
            self.assertGreater(probs[0], probs[i])

    def test_value_arm0_dominates(self):
        self.bandit.observe_data(0, 90, 100)
        self.bandit.observe_data(1, 30, 100)
        self.bandit.observe_data(2, 20, 100)
        self.bandit.observe_data(3, 10, 100)
        self.bandit.update_posterior(500)
        v0 = self.bandit.value(0)
        for i in range(1, 4):
            self.assertGreater(v0, self.bandit.value(i))

    def test_value_remaining_distribution_length(self):
        self.bandit.observe_data(0, 5, 10)
        self.bandit.observe_data(1, 2, 10)
        self.bandit.update_posterior(200)
        vr = self.bandit.value_remaining_distribution()
        self.assertEqual(200, len(vr))

    def test_arm_predictors_shape(self):
        self.bandit.observe_data(0, 5, 10)
        self.bandit.update_posterior(10)
        pred = self.bandit.arm_predictors()
        # 4 arms, intercept(1) + ButtonPosition(1) + ButtonColor(1) = 3 cols
        self.assertEqual(4, pred.shape[0])
        self.assertEqual(3, pred.shape[1])

    def test_arm_predictors_intercept_column(self):
        self.bandit.observe_data(0, 5, 10)
        self.bandit.update_posterior(10)
        pred = self.bandit.arm_predictors()
        for i in range(pred.shape[0]):
            self.assertAlmostEqual(1.0, pred[i, 0])

    def test_arm_predictors_rows_distinct(self):
        self.bandit.observe_data(0, 5, 10)
        self.bandit.update_posterior(10)
        pred = self.bandit.arm_predictors()
        for i in range(pred.shape[0]):
            for j in range(i + 1, pred.shape[0]):
                self.assertFalse(np.allclose(pred[i], pred[j]))

    def test_observe_after_boom_initialized(self):
        # Adding data after the first update_posterior resets the boom objects
        # so the next update_posterior rebuilds from all training data.
        self.bandit.observe_data(0, 3, 5)
        self.bandit.update_posterior(10)
        self.bandit.observe_data(1, 1, 5)  # resets _boom_bandit
        self.bandit.update_posterior(10)   # rebuilds and resamples
        self.assertEqual(10, self.bandit.ndraws)


class TestLogitBanditWithContext(unittest.TestCase):

    def setUp(self):
        self.bandit = _make_bandit_with_context()

    def _ctx(self, x1_value):
        return pd.DataFrame({"x1": [float(x1_value)]})

    def test_update_posterior_with_context(self):
        ctx = self._ctx(1.5)
        self.bandit.observe_data(0, 5, 10, ctx)
        self.bandit.observe_data(1, 2, 10, ctx)
        self.assertEqual(0, self.bandit.ndraws)
        self.bandit.update_posterior(200)
        self.assertEqual(200, self.bandit.ndraws)

    def test_optimal_arm_probs_with_context_sums_to_one(self):
        ctx = self._ctx(1.5)
        self.bandit.observe_data(0, 90, 100, ctx)
        self.bandit.observe_data(1, 30, 100, ctx)
        self.bandit.observe_data(2, 20, 100, ctx)
        self.bandit.observe_data(3, 10, 100, ctx)
        self.bandit.update_posterior(1000)
        probs = self.bandit.optimal_arm_probabilities(ctx)
        self.assertEqual(4, len(probs))
        self.assertAlmostEqual(1.0, probs.sum(), places=10)

    def test_optimal_arm_probs_with_context_arm0_dominates(self):
        ctx = self._ctx(1.5)
        self.bandit.observe_data(0, 90, 100, ctx)
        self.bandit.observe_data(1, 30, 100, ctx)
        self.bandit.observe_data(2, 20, 100, ctx)
        self.bandit.observe_data(3, 10, 100, ctx)
        self.bandit.update_posterior(1000)
        probs = self.bandit.optimal_arm_probabilities(ctx)
        for i in range(1, 4):
            self.assertGreater(probs[0], probs[i])

    def test_value_with_context(self):
        ctx = self._ctx(1.5)
        self.bandit.observe_data(0, 90, 100, ctx)
        self.bandit.observe_data(1, 10, 100, ctx)
        self.bandit.update_posterior(500)
        self.assertGreater(self.bandit.value(0, ctx), self.bandit.value(1, ctx))

    def test_arm_predictors_with_context_shape(self):
        ctx = self._ctx(1.5)
        self.bandit.observe_data(0, 5, 10, ctx)
        self.bandit.update_posterior(10)
        pred = self.bandit.arm_predictors(ctx)
        # 4 arms, intercept(1) + ButtonPosition(1) + ButtonColor(1) + x1(1) = 4 cols
        self.assertEqual(4, pred.shape[0])
        self.assertEqual(4, pred.shape[1])


def _make_bandit_with_external_value(value_function):
    """2-factor experiment with a custom value function, no context."""
    xp = ExperimentStructure()
    xp.add_factor("ButtonPosition", ["Left", "Right"])
    xp.add_factor("ButtonColor", ["Red", "Blue"])
    arm_map = ArmMap(xp)
    pos_enc = ExperimentArmEncoder("ButtonPosition", arm_map)
    col_enc = ExperimentArmEncoder("ButtonColor", arm_map)
    dataset_encoder = R.DatasetEncoder([pos_enc, col_enc])
    encoder = LinearBanditEncoder(arm_map, dataset_encoder)
    return LogitBandit(arm_map, encoder, value_function=value_function)


def _position_weighted(prob, arm_levels):
    """Value = 2 * prob for Left arms, prob for Right arms.

    arm_levels are "FactorName:LevelName" strings, so ButtonPosition:Left vs
    ButtonPosition:Right.
    """
    return prob * (2.0 if arm_levels[0] == "ButtonPosition:Left" else 1.0)


# Arm ordering for ButtonPosition x ButtonColor (2x2):
#   arm 0: ButtonPosition:Left,  ButtonColor:Red
#   arm 1: ButtonPosition:Left,  ButtonColor:Blue
#   arm 2: ButtonPosition:Right, ButtonColor:Red
#   arm 3: ButtonPosition:Right, ButtonColor:Blue


class TestLogitBanditExternalValue(unittest.TestCase):

    def setUp(self):
        self.bandit = _make_bandit_with_external_value(_position_weighted)

    def test_boom_object_is_external_value_type(self):
        import BayesBoom.boom as boom
        self.bandit.observe_data(0, 5, 10)
        self.assertIsInstance(self.bandit.boom(), boom.LogitBanditExternalValue)

    def test_plain_bandit_is_not_external_value_type(self):
        import BayesBoom.boom as boom
        plain = _make_bandit_no_context()
        plain.observe_data(0, 5, 10)
        self.assertIsInstance(plain.boom(), boom.LogitBandit)
        self.assertNotIsInstance(plain.boom(), boom.LogitBanditExternalValue)

    def test_value_scaled_by_value_function(self):
        # With zero initial coefficients, logit_inv(0) = 0.5 for every arm.
        # The value function doubles Left arms, so the ratio is exactly 2.0.
        for left_arm, right_arm in [(0, 2), (1, 3)]:
            ratio = self.bandit.value(left_arm) / self.bandit.value(right_arm)
            self.assertAlmostEqual(2.0, ratio, places=10)

    def test_value_function_receives_arm_levels_for_left_red(self):
        # Arm 0 is (ButtonPosition:Left, ButtonColor:Red).
        captured = []
        def capturing(prob, arm_levels):
            captured.append(list(arm_levels))
            return prob

        bandit = _make_bandit_with_external_value(capturing)
        bandit.observe_data(0, 5, 10)
        bandit.update_posterior(10)
        bandit.value(0)
        self.assertEqual(1, len(captured))
        self.assertEqual(["ButtonPosition:Left", "ButtonColor:Red"], captured[0])

    def test_value_function_receives_arm_levels_for_right_red(self):
        # Arm 2 is (ButtonPosition:Right, ButtonColor:Red).
        captured = []
        def capturing(prob, arm_levels):
            captured.append(list(arm_levels))
            return prob

        bandit = _make_bandit_with_external_value(capturing)
        bandit.observe_data(2, 5, 10)
        bandit.update_posterior(10)
        bandit.value(2)
        self.assertEqual(1, len(captured))
        self.assertEqual(["ButtonPosition:Right", "ButtonColor:Red"], captured[0])

    def test_optimal_arm_probs_sum_to_one(self):
        for arm in range(4):
            self.bandit.observe_data(arm, 50, 100)
        self.bandit.update_posterior(500)
        probs = self.bandit.optimal_arm_probabilities()
        self.assertEqual(4, len(probs))
        self.assertAlmostEqual(1.0, probs.sum(), places=10)

    def test_optimal_arm_probs_reflect_value_function(self):
        # Arm 2 (Right/Red) is the raw probability winner.  The value function
        # doubles Left arms, so Left arms should collectively be seen as more
        # likely optimal than Right arms: value(Left, ~0.5)*2 > value(Right, ~0.9).
        self.bandit.observe_data(0, 50, 100)   # Left/Red
        self.bandit.observe_data(1, 50, 100)   # Left/Blue
        self.bandit.observe_data(2, 90, 100)   # Right/Red <- raw winner
        self.bandit.observe_data(3, 50, 100)   # Right/Blue
        self.bandit.update_posterior(1000)
        probs = self.bandit.optimal_arm_probabilities()
        self.assertGreater(probs[0] + probs[1], probs[2] + probs[3])

    def test_value_remaining_distribution_length(self):
        for arm in range(4):
            self.bandit.observe_data(arm, 50, 100)
        self.bandit.update_posterior(200)
        vr = self.bandit.value_remaining_distribution()
        self.assertEqual(200, len(vr))

    def test_value_remaining_distribution_near_zero_when_arm0_dominates(self):
        # Value function gives arm 0 a 10x multiplier; it will win almost every
        # posterior draw, so best_value - arm0_value should be ~0 in each draw.
        def arm0_wins(prob, arm_levels):
            return prob * (10.0 if arm_levels == ["ButtonPosition:Left",
                                                  "ButtonColor:Red"] else 1.0)

        bandit = _make_bandit_with_external_value(arm0_wins)
        for arm in range(4):
            bandit.observe_data(arm, 50, 100)
        bandit.update_posterior(500)
        vr = bandit.value_remaining_distribution()
        self.assertTrue(np.all(vr <= 1e-6))


class TestLinearBanditEncoderDim(unittest.TestCase):

    def test_dim_no_context(self):
        xp = ExperimentStructure()
        xp.add_factor("A", ["a1", "a2"])
        xp.add_factor("B", ["b1", "b2", "b3"])
        arm_map = ArmMap(xp)
        a_enc = ExperimentArmEncoder("A", arm_map)
        b_enc = ExperimentArmEncoder("B", arm_map)
        dataset_encoder = R.DatasetEncoder([a_enc, b_enc])
        encoder = LinearBanditEncoder(arm_map, dataset_encoder)
        # ExperimentArmEncoder.dim = nfactors - 1 = 1 for each encoder in a
        # 2-factor experiment.  Total: intercept(1) + A(1) + B(2) = 4.
        self.assertEqual(4, encoder.dim)

    def test_number_of_arms(self):
        xp = ExperimentStructure()
        xp.add_factor("A", ["a1", "a2"])
        xp.add_factor("B", ["b1", "b2", "b3"])
        arm_map = ArmMap(xp)
        a_enc = ExperimentArmEncoder("A", arm_map)
        b_enc = ExperimentArmEncoder("B", arm_map)
        dataset_encoder = R.DatasetEncoder([a_enc, b_enc])
        encoder = LinearBanditEncoder(arm_map, dataset_encoder)
        # 2 x 3 = 6 arms
        self.assertEqual(6, encoder.number_of_arms)

    def test_encode_row_length(self):
        xp = ExperimentStructure()
        xp.add_factor("A", ["a1", "a2"])
        arm_map = ArmMap(xp)
        a_enc = ExperimentArmEncoder("A", arm_map)
        dataset_encoder = R.DatasetEncoder([a_enc])
        encoder = LinearBanditEncoder(arm_map, dataset_encoder)
        row = encoder.encode_row(0, None)
        self.assertEqual(encoder.dim, len(row))


if __name__ == "__main__":
    unittest.main()

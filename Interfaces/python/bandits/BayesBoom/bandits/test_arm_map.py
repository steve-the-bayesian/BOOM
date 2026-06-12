import unittest
import numpy as np

from BayesBoom.bandits.linear_bandit_encoder import ExperimentStructure, ArmMap


def _make_experiment(rates=None, limits=None):
    xp = ExperimentStructure()
    xp.add_factor("InterestRate", rates or ["0.0", "0.1", "0.2", "0.3"])
    xp.add_factor("CreditLimit", limits or ["0", "10", "20", "50", "100"])
    return xp


class TestArmMap(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    # -----------------------------------------------------------------------
    # number_of_arms
    # -----------------------------------------------------------------------

    def test_number_of_arms_two_factor(self):
        arm_map = ArmMap(_make_experiment())
        # 4 interest rates x 5 credit limits
        self.assertEqual(20, arm_map.number_of_arms)

    def test_number_of_arms_single_factor(self):
        xp = ExperimentStructure()
        xp.add_factor("Color", ["Red", "Green", "Blue"])
        arm_map = ArmMap(xp)
        self.assertEqual(3, arm_map.number_of_arms)

    def test_number_of_arms_three_factors(self):
        xp = ExperimentStructure()
        xp.add_factor("A", ["a1", "a2"])
        xp.add_factor("B", ["b1", "b2", "b3"])
        xp.add_factor("C", ["c1", "c2"])
        arm_map = ArmMap(xp)
        self.assertEqual(12, arm_map.number_of_arms)

    # -----------------------------------------------------------------------
    # map shape and values
    # -----------------------------------------------------------------------

    def test_map_shape(self):
        arm_map = ArmMap(_make_experiment())
        # (number_of_arms, number_of_factors)
        self.assertEqual((20, 2), arm_map.map.shape)

    def test_map_shape_single_factor(self):
        xp = ExperimentStructure()
        xp.add_factor("Color", ["Red", "Green", "Blue"])
        self.assertEqual((3, 1), ArmMap(xp).map.shape)

    def test_map_factor0_values_in_range(self):
        arm_map = ArmMap(_make_experiment())
        col = arm_map.map[:, 0]  # InterestRate: 4 levels
        self.assertTrue(np.all(col >= 0))
        self.assertTrue(np.all(col < 4))

    def test_map_factor1_values_in_range(self):
        arm_map = ArmMap(_make_experiment())
        col = arm_map.map[:, 1]  # CreditLimit: 5 levels
        self.assertTrue(np.all(col >= 0))
        self.assertTrue(np.all(col < 5))

    def test_map_rows_are_distinct(self):
        arm_map = ArmMap(_make_experiment())
        rows = [tuple(row) for row in arm_map.map]
        self.assertEqual(len(rows), len(set(rows)))

    def test_map_covers_all_combinations(self):
        # Every combination of factor levels must appear exactly once.
        arm_map = ArmMap(_make_experiment())
        rows = set(tuple(row) for row in arm_map.map.astype(int))
        expected = {(r, c) for r in range(4) for c in range(5)}
        self.assertEqual(expected, rows)

    # -----------------------------------------------------------------------
    # boom factor_level_names and integer_factor_levels
    # -----------------------------------------------------------------------

    def test_factor_level_names_format(self):
        arm_map = ArmMap(_make_experiment())
        bam = arm_map.boom()
        for arm in range(arm_map.number_of_arms):
            names = bam.factor_level_names(arm)
            self.assertEqual(2, len(names))
            self.assertTrue(names[0].startswith("InterestRate:"))
            self.assertTrue(names[1].startswith("CreditLimit:"))

    def test_factor_level_names_values(self):
        arm_map = ArmMap(_make_experiment())
        bam = arm_map.boom()
        flm = arm_map.map.astype(int)
        rates = ["0.0", "0.1", "0.2", "0.3"]
        limits = ["0", "10", "20", "50", "100"]
        for arm in range(arm_map.number_of_arms):
            names = bam.factor_level_names(arm)
            r_idx, c_idx = flm[arm, 0], flm[arm, 1]
            self.assertEqual(f"InterestRate:{rates[r_idx]}", names[0])
            self.assertEqual(f"CreditLimit:{limits[c_idx]}", names[1])

    def test_integer_factor_levels_match_map(self):
        arm_map = ArmMap(_make_experiment())
        bam = arm_map.boom()
        flm = arm_map.map.astype(int)
        for arm in range(arm_map.number_of_arms):
            int_levels = bam.integer_factor_levels(arm)
            self.assertEqual(int(flm[arm, 0]), int_levels[0])
            self.assertEqual(int(flm[arm, 1]), int_levels[1])

    # -----------------------------------------------------------------------
    # map matches boom factor_level_matrix
    # -----------------------------------------------------------------------

    def test_python_map_equals_boom_factor_level_matrix(self):
        arm_map = ArmMap(_make_experiment())
        import BayesBoom.R as R
        boom_matrix = R.to_numpy(arm_map.boom().factor_level_matrix)
        np.testing.assert_array_equal(arm_map.map, boom_matrix)


if __name__ == "__main__":
    unittest.main()

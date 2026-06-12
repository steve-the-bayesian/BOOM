import unittest
import numpy as np

from BayesBoom.bandits.linear_bandit_encoder import ExperimentStructure


class TestExperimentStructure(unittest.TestCase):

    def setUp(self):
        np.random.seed(8675309)

    def _make_two_factor_experiment(self):
        xp = ExperimentStructure()
        xp.add_factor("InterestRate", ["0.0", "0.1", "0.2", "0.3"])
        xp.add_factor("CreditLimit", ["0", "10", "20", "50", "100"])
        return xp

    # -----------------------------------------------------------------------
    # Python-side __getitem__
    # -----------------------------------------------------------------------

    def test_getitem_first_factor(self):
        xp = self._make_two_factor_experiment()
        self.assertEqual(["0.0", "0.1", "0.2", "0.3"], xp["InterestRate"])

    def test_getitem_second_factor(self):
        xp = self._make_two_factor_experiment()
        self.assertEqual(["0", "10", "20", "50", "100"], xp["CreditLimit"])

    def test_factor_names_list(self):
        xp = self._make_two_factor_experiment()
        self.assertEqual(["InterestRate", "CreditLimit"], xp._factor_names)

    def test_factor_levels_list(self):
        xp = self._make_two_factor_experiment()
        self.assertEqual(
            [["0.0", "0.1", "0.2", "0.3"], ["0", "10", "20", "50", "100"]],
            xp._factor_levels)

    def test_single_factor(self):
        xp = ExperimentStructure()
        xp.add_factor("Color", ["Red", "Green", "Blue"])
        self.assertEqual(["Color"], xp._factor_names)
        self.assertEqual([["Red", "Green", "Blue"]], xp._factor_levels)
        self.assertEqual(["Red", "Green", "Blue"], xp["Color"])

    # -----------------------------------------------------------------------
    # boom() agreement
    # -----------------------------------------------------------------------

    def test_boom_levels_match_python(self):
        xp = self._make_two_factor_experiment()
        bxp = xp.boom()
        self.assertEqual(bxp.levels("InterestRate"), xp["InterestRate"])
        self.assertEqual(bxp.levels("CreditLimit"), xp["CreditLimit"])

    def test_boom_factor_names_match_python(self):
        xp = self._make_two_factor_experiment()
        bxp = xp.boom()
        self.assertEqual(bxp.factor_names, xp._factor_names)

    def test_boom_nfactors(self):
        xp = self._make_two_factor_experiment()
        bxp = xp.boom()
        self.assertEqual(2, bxp.nfactors)

    def test_boom_nconfigurations(self):
        xp = self._make_two_factor_experiment()
        bxp = xp.boom()
        # 4 interest rates x 5 credit limits = 20
        self.assertEqual(20, bxp.nconfigurations)

    def test_boom_nconfigurations_single_factor(self):
        xp = ExperimentStructure()
        xp.add_factor("A", ["a1", "a2", "a3"])
        self.assertEqual(3, xp.boom().nconfigurations)

    def test_boom_nconfigurations_three_factors(self):
        xp = ExperimentStructure()
        xp.add_factor("A", ["a1", "a2"])
        xp.add_factor("B", ["b1", "b2", "b3"])
        xp.add_factor("C", ["c1", "c2"])
        # 2 * 3 * 2 = 12
        self.assertEqual(12, xp.boom().nconfigurations)

    # -----------------------------------------------------------------------
    # Lazy boom initialization
    # -----------------------------------------------------------------------

    def test_boom_lazy_init_same_object(self):
        xp = self._make_two_factor_experiment()
        bxp1 = xp.boom()
        bxp2 = xp.boom()
        self.assertIs(bxp1, bxp2)

    def test_boom_not_created_before_first_call(self):
        xp = ExperimentStructure()
        self.assertIsNone(xp._boom_experiment_structure)
        xp.add_factor("A", ["a1", "a2"])
        self.assertIsNone(xp._boom_experiment_structure)
        xp.boom()
        self.assertIsNotNone(xp._boom_experiment_structure)

    # -----------------------------------------------------------------------
    # add_factor after boom() keeps both in sync
    # -----------------------------------------------------------------------

    def test_add_factor_after_boom_updates_boom_nfactors(self):
        xp = ExperimentStructure()
        xp.add_factor("InterestRate", ["0.0", "0.1"])
        bxp = xp.boom()
        self.assertEqual(1, bxp.nfactors)
        xp.add_factor("CreditLimit", ["0", "10", "20"])
        self.assertEqual(2, bxp.nfactors)

    def test_add_factor_after_boom_updates_boom_levels(self):
        xp = ExperimentStructure()
        xp.add_factor("A", ["a1", "a2"])
        bxp = xp.boom()
        xp.add_factor("B", ["b1", "b2", "b3"])
        self.assertEqual(["b1", "b2", "b3"], bxp.levels("B"))


if __name__ == "__main__":
    unittest.main()

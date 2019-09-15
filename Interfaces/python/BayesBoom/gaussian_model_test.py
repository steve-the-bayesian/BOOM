import unittest
import BayesBoom as boom
import numpy as np

class GaussianModelTest(unittest.TestCase):

    def setUp(self):
        self.data = np.array([1, 2, 3])

    def test_moments(self):
        model = boom.GaussianModel(1, 2)
        self.assertEqual(1.0, model.mean())
        self.assertAlmostEqual(2.0, model.sd())
        self.assertAlmostEqual(4.0, model.variance())
        
if __name__ == "__main__":
    unittest.main()

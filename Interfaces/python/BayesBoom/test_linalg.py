import unittest
import BayesBoom as boom
import numpy as np

class VectorTest(unittest.TestCase):

    def setUp(self):
        self.data = np.array([1, 2, 5])

    def test_constructors(self):
        v1 = boom.Vector(18, 12)
        self.assertEqual(v1.size(), 18)

        v2 = boom.Vector(self.data)
        
        
if __name__ == "__main__":
    unittest.main()

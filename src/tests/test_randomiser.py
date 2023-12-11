import unittest
from framework import Randomiser


class TestRandomiser(unittest.TestCase):
    def test_rand_seed(self):
        self.assertEqual(Randomiser.get_rand_seed(1), 2267)

    def test_init_guess(self):
        x, y = Randomiser.get_init_guess(2)
        self.assertAlmostEqual(x, 151509.5802093)
        self.assertAlmostEqual(y, 399956.21836103)

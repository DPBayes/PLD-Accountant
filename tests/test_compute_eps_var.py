
'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

Tests for epsilon computation with varying sigma and q

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''


import unittest
import numpy as np

from fourier_accountant.compute_eps import get_epsilon_R, get_epsilon_S

class compute_epsilon_regression_tests(unittest.TestCase):

    def test_get_epsilon_R_regression_valid_params(self):
        """ Tests that results of get_epsilon_R did not drift from last version."""
        nc = 10
        sigmas = np.linspace(1.6, 1.8, nc)
        q_values = np.linspace(0.05, 0.06, nc)
        ks = 10 * np.ones(nc, dtype=np.int32) #number of compositions for each value of (q,sigma)

        test_data = [
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_delta=1e-3, nx=1E6, L=20.0), 0.9658229984720059),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_delta=1e-3, nx=1E6, L=40.0), 0.9658230049986837),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_delta=1e-3, nx=1E5, L=20.0), 0.9658230533746907),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_delta=1e-4, nx=1E6, L=20.0), 1.2658081375076573),
            (dict(sigma=sigmas[:-1], q=q_values[:-1], ncomp=ks[:-1], target_delta=1e-3, nx=1E6, L=20.0), 0.9092052395489234),
        ]

        for params, expected in test_data:
            actual = get_epsilon_R(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_epsilon_R_invalid_sizes(self):
        with self.assertRaises(ValueError):
            get_epsilon_R(sigma = np.ones(2), q = np.ones(2), ncomp = np.ones(1, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_epsilon_R(sigma = np.ones(2), q = np.ones(1), ncomp = np.ones(2, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_epsilon_R(sigma = np.ones(1), q = np.ones(2), ncomp = np.ones(2, dtype=np.int32))


    def test_get_epsilon_R_exceptions(self):
        """ Tests that get_epsilon_R raises errors when encountering instabilities."""
        nc = 10
        ks = 10 * np.ones(nc, dtype=np.int32) #number of compositions for each value of (q,sigma)

        with self.assertRaises(ValueError):
            get_epsilon_R(sigma = np.linspace(0.0005, 0.0015, nc), q = np.linspace(0.15, 0.25, nc), ncomp = ks, target_delta=1e-4, nx=1E6, L=40.0)


    def test_get_epsilon_S_regression_valid_params(self):
        """ Tests that results of get_epsilon_S did not drift from last version."""
        nc = 10
        sigmas = np.linspace(1.6, 1.8, nc)
        q_values = np.linspace(0.05, 0.06, nc)
        ks = 10 * np.ones(nc, dtype=np.int32) #number of compositions for each value of (q,sigma)

        test_data = [
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_delta=1e-3, nx=1E6, L=20.0), 1.849343750228688),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_delta=1e-3, nx=1E6, L=40.0), 1.8493437498330085),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_delta=1e-3, nx=1E5, L=20.0), 1.8493437767177145),
            (dict(sigma=sigmas[:-1], q=q_values[:-1], ncomp=ks[:-1], target_delta=1e-4, nx=1E6, L=20.0), 2.157488007444617),
        ]

        for params, expected in test_data:
            actual = get_epsilon_S(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_epsilon_S_invalid_sizes(self):
        with self.assertRaises(ValueError):
            get_epsilon_S(sigma = np.ones(2), q = np.ones(2), ncomp = np.ones(1, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_epsilon_S(sigma = np.ones(2), q = np.ones(1), ncomp = np.ones(2, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_epsilon_S(sigma = np.ones(1), q = np.ones(2), ncomp = np.ones(2, dtype=np.int32))


    def test_get_epsilon_S_exceptions(self):
        """ Tests that get_epsilon_S raises errors when encountering instabilities."""
        nc = 10
        ks = 10 * np.ones(nc, dtype=np.int32) #number of compositions for each value of (q,sigma)

        with self.assertRaises(ValueError):
            get_epsilon_S(sigma = np.linspace(0.0005, 0.0015, nc), q = np.linspace(0.15, 0.25, nc), ncomp = ks, target_delta=1e-4, nx=1E6, L=40.0)

    def test_get_epsilon_enforces_all_array_or_all_scalar(self):
        with self.assertRaises(TypeError):
            get_epsilon_S(sigma = 1, q = np.ones(1), ncomp = 10)
        with self.assertRaises(TypeError):
            get_epsilon_R(sigma = 1, q = np.ones(1), ncomp = 10)

if __name__ == '__main__':
    unittest.main()

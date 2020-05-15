
'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

Tests for vector variant of epsilon computation

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''


import unittest
import numpy as np

from fourier_accountant.compute_eps_var import get_epsilon_R, get_epsilon_S

class compute_epsilon_regression_tests(unittest.TestCase):

    def test_get_epsilon_R_regression_valid_params(self):
        """ Tests that results of get_epsilon_R did not drift from last version."""
        nc = 10
        sigmas = np.linspace(1.6, 1.8, nc)
        q_values = np.linspace(0.05, 0.06, nc)
        ks = 10 * np.ones(nc) #number of compositions for each value of (q,sigma)

        test_data = [
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_delta=1e-3, nx=1E6, L=20.0), 0.9658229984720059),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_delta=1e-3, nx=1E6, L=40.0), 0.9658230049986837),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_delta=1e-3, nx=1E5, L=20.0), 0.9658230533746907),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_delta=1e-4, nx=1E6, L=20.0), 1.2658081375076573),
            (dict(sigma_t=sigmas[:-1], q_t=q_values[:-1], k=ks[:-1], target_delta=1e-3, nx=1E6, L=20.0), 0.9092052395489234),
        ]

        for params, expected in test_data:
            actual = get_epsilon_R(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_epsilon_R_invalid_sizes(self):
        with self.assertRaises(ValueError):
            get_epsilon_R(sigma_t = np.ones(2), q_t = np.ones(2), k = np.ones(1))
            get_epsilon_R(sigma_t = np.ones(2), q_t = np.ones(1), k = np.ones(2))
            get_epsilon_R(sigma_t = np.ones(1), q_t = np.ones(2), k = np.ones(2))


    def test_get_epsilon_R_exceptions(self):
        """ Tests that get_epsilon_R raises errors when encountering instabilities."""
        nc = 10
        ks = 10 * np.ones(nc) #number of compositions for each value of (q,sigma)

        with self.assertRaises(ValueError):
            get_epsilon_R(sigma_t = np.linspace(0.45, 0.55, nc), q_t = np.linspace(0.01, 0.02, nc), k = ks, target_delta=1e-6, nx=1E6, L=5.0)
            get_epsilon_R(sigma_t = np.linspace(0.0005, 0.0015, nc), q_t = np.linspace(0.15, 0.25, nc), k = ks, target_delta=1e-4, nx=1E6, L=40.0)



    def test_get_epsilon_S_regression_valid_params(self):
        """ Tests that results of get_epsilon_S did not drift from last version."""
        nc = 10
        sigmas = np.linspace(1.6, 1.8, nc)
        q_values = np.linspace(0.05, 0.06, nc)
        ks = 10 * np.ones(nc) #number of compositions for each value of (q,sigma)

        test_data = [
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_delta=1e-3, nx=1E6, L=20.0), 1.849343750228688),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_delta=1e-3, nx=1E6, L=40.0), 1.8493437498330085),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_delta=1e-3, nx=1E5, L=20.0), 1.8493437767177145),
            (dict(sigma_t=sigmas[:-1], q_t=q_values[:-1], k=ks[:-1], target_delta=1e-4, nx=1E6, L=20.0), 2.157488007444617),
        ]

        for params, expected in test_data:
            actual = get_epsilon_S(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_epsilon_S_invalid_sizes(self):
        with self.assertRaises(ValueError):
            get_epsilon_S(sigma_t = np.ones(2), q_t = np.ones(2), k = np.ones(1))
            get_epsilon_S(sigma_t = np.ones(2), q_t = np.ones(1), k = np.ones(2))
            get_epsilon_S(sigma_t = np.ones(1), q_t = np.ones(2), k = np.ones(2))


    def test_get_epsilon_S_exceptions(self):
        """ Tests that get_epsilon_S raises errors when encountering instabilities."""
        nc = 10
        ks = 10 * np.ones(nc) #number of compositions for each value of (q,sigma)

        with self.assertRaises(ValueError):
            get_epsilon_S(sigma_t = np.linspace(0.45, 0.55, nc), q_t = np.linspace(0.01, 0.02, nc), k = ks, target_delta=1e-6, nx=1E6, L=5.0)
            get_epsilon_S(sigma_t = np.linspace(0.0005, 0.0015, nc), q_t = np.linspace(0.15, 0.25, nc), k = ks, target_delta=1e-4, nx=1E6, L=40.0)



if __name__ == '__main__':
    unittest.main()

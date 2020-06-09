
'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

Tests for delta computation with varying sigma and q through old interface

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''


import unittest
import numpy as np

from fourier_accountant.compute_delta_var import get_delta_R, get_delta_S

class compute_delta_regression_tests(unittest.TestCase):

    def test_get_delta_R_regression_valid_params(self):
        """ Tests that results of get_delta_R did not drift from last version."""
        nc = 10
        sigmas = np.linspace(1.6, 1.8, nc)
        q_values = np.linspace(0.05, 0.06, nc)
        ks = 10 * np.ones(nc, dtype=np.int32) #number of compositions for each value of (q,sigma)

        test_data = [
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_eps=1.0, nx=1E6, L=20.0), 0.0007824006722393885),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_eps=1.0, nx=1E6, L=40.0), 0.0007824006547443852),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_eps=1.0, nx=1E5, L=20.0), 0.0007824000947941229),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_eps=0.8, nx=1E6, L=20.0), 0.0030814519279194538),
            (dict(sigma_t=sigmas[:-1], q_t=q_values[:-1], k=ks[:-1], target_eps=1.0, nx=1E6, L=20.0), 0.0005014987387132376),
        ]

        for params, expected in test_data:
            actual = get_delta_R(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_delta_R_invalid_sizes(self):
        with self.assertRaises(ValueError):
            get_delta_R(sigma_t = np.ones(2), q_t = np.ones(2), k = np.ones(1, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_delta_R(sigma_t = np.ones(2), q_t = np.ones(1), k = np.ones(2, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_delta_R(sigma_t = np.ones(1), q_t = np.ones(2), k = np.ones(2, dtype=np.int32))

    def test_get_delta_S_regression_valid_params(self):
        """ Tests that results of get_delta_S did not drift from last version."""
        nc = 10
        sigmas = np.linspace(1.6, 1.8, nc)
        q_values = np.linspace(0.05, 0.06, nc)
        ks = 10 * np.ones(nc, dtype=np.int32) #number of compositions for each value of (q,sigma)

        test_data = [
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_eps=1.0, nx=1E6, L=20.0), 0.026891238748125684),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_eps=1.0, nx=1E6, L=40.0), 0.026891238632122155),
            (dict(sigma_t=sigmas, q_t=q_values, k=ks, target_eps=1.0, nx=1E5, L=20.0), 0.026891234920482492),
            (dict(sigma_t=sigmas[:-1], q_t=q_values[:-1], k=ks[:-1], target_eps=0.8, nx=1E6, L=20.0), 0.039256512455315),
        ]

        for params, expected in test_data:
            actual = get_delta_S(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_delta_S_invalid_sizes(self):
        with self.assertRaises(ValueError):
            get_delta_S(sigma_t = np.ones(2), q_t = np.ones(2), k = np.ones(1, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_delta_S(sigma_t = np.ones(2), q_t = np.ones(1), k = np.ones(2, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_delta_S(sigma_t = np.ones(1), q_t = np.ones(2), k = np.ones(2, dtype=np.int32))


if __name__ == '__main__':
    unittest.main()

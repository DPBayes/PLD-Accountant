
'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

Tests for delta computation with varying sigma and q

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''


import unittest
import numpy as np

from fourier_accountant.compute_delta import get_delta_R, get_delta_S

class compute_delta_regression_tests(unittest.TestCase):

    def test_get_delta_R_regression_valid_params(self):
        """ Tests that results of get_delta_R did not drift from last version."""
        nc = 10
        sigmas = np.linspace(1.6, 1.8, nc)
        q_values = np.linspace(0.05, 0.06, nc)
        ks = 10 * np.ones(nc, dtype=np.int32) #number of compositions for each value of (q,sigma)

        test_data = [
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_eps=1.0, nx=1E6, L=20.0), 0.0007824006722393885),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_eps=1.0, nx=1E6, L=40.0), 0.0007824006547443852),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_eps=1.0, nx=1E5, L=20.0), 0.0007824000947941229),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_eps=0.8, nx=1E6, L=20.0), 0.0030814519279194538),
            (dict(sigma=sigmas[:-1], q=q_values[:-1], ncomp=ks[:-1], target_eps=1.0, nx=1E6, L=20.0), 0.0005014987387132376),
        ]

        for params, expected in test_data:
            actual = get_delta_R(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_delta_R_invalid_sizes(self):
        with self.assertRaises(ValueError):
            get_delta_R(sigma = np.ones(2), q = np.ones(2), ncomp = np.ones(1, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_delta_R(sigma = np.ones(2), q = np.ones(1), ncomp = np.ones(2, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_delta_R(sigma = np.ones(1), q = np.ones(2), ncomp = np.ones(2, dtype=np.int32))

    def test_get_delta_S_regression_valid_params(self):
        """ Tests that results of get_delta_S did not drift from last version."""
        nc = 10
        sigmas = np.linspace(1.6, 1.8, nc)
        q_values = np.linspace(0.05, 0.06, nc)
        ks = 10 * np.ones(nc, dtype=np.int32) #number of compositions for each value of (q,sigma)

        test_data = [
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_eps=1.0, nx=1E6, L=20.0), 0.026891238748125684),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_eps=1.0, nx=1E6, L=40.0), 0.026891238632122155),
            (dict(sigma=sigmas, q=q_values, ncomp=ks, target_eps=1.0, nx=1E5, L=20.0), 0.026891234920482492),
            (dict(sigma=sigmas[:-1], q=q_values[:-1], ncomp=ks[:-1], target_eps=0.8, nx=1E6, L=20.0), 0.039256512455315),
        ]

        for params, expected in test_data:
            actual = get_delta_S(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_delta_S_invalid_sizes(self):
        with self.assertRaises(ValueError):
            get_delta_S(sigma = np.ones(2), q = np.ones(2), ncomp = np.ones(1, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_delta_S(sigma = np.ones(2), q = np.ones(1), ncomp = np.ones(2, dtype=np.int32))
        with self.assertRaises(ValueError):
            get_delta_S(sigma = np.ones(1), q = np.ones(2), ncomp = np.ones(2, dtype=np.int32))

    def test_get_delta_enforces_all_array_or_all_scalar(self):
        with self.assertRaises(TypeError):
            get_delta_S(sigma = 1, q = np.ones(1), ncomp = 10)
        with self.assertRaises(TypeError):
            get_delta_R(sigma = 1, q = np.ones(1), ncomp = 10)


if __name__ == '__main__':
    unittest.main()

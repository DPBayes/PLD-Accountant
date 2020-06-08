
'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

Tests for epsilon computation with constant sigma and q

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''

import unittest

from fourier_accountant import get_epsilon_R, get_epsilon_S

class compute_eps_regression_tests(unittest.TestCase):

    def test_get_epsilon_R_regression_valid_params(self):
        """ Tests that results of get_epsilon_R did not drift from last version."""
        test_data = [
            (dict(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0),  2.446734047007243),
            (dict(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=40.0),  2.446733916332568),
            (dict(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E5, L=20.0),  2.4467323179602007),
            (dict(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E5, nx=1E6, L=20.0),  8.984848512304165),
            (dict(target_delta=1e-5, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0),  2.162704603256141),
            (dict(target_delta=1e-6, sigma=1.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0),  6.907422808389587),
            (dict(target_delta=1e-6, sigma=1.0, q=0.02, ncomp=1E4, nx=1E6, L=20.0), 15.64135722007042),
        ]

        for params, expected in test_data:
            actual = get_epsilon_R(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_epsilon_R_exceptions(self):
        """ Tests that get_epsilon_R raises errors when encountering instabilities."""
        with self.assertRaises(ValueError):
            get_epsilon_R(target_delta=1e-6, sigma=.5, q=0.01, ncomp=1E4, nx=1E6, L=5.0)
            get_epsilon_R(target_delta=1e-4, sigma=0.001, q=0.2, ncomp=1E4, nx=1E6, L=40.0)

    def test_get_epsilon_S_regression_valid_params(self):
        """ Tests that results of get_epsilon_S did not drift from last version."""
        test_data = [
            (dict(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0),  4.907397768991888),
            (dict(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=40.0),  4.907397523232251),
            (dict(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E5, L=20.0),  4.9073978692869344),
            (dict(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E5, nx=1E6, L=20.0), 19.0867527960459),
            (dict(target_delta=1e-5, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0),  4.396030629364786),
            (dict(target_delta=1e-6, sigma=1.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0), 11.956857696159513),
            (dict(target_delta=1e-6, sigma=2.0, q=0.005, ncomp=1E4, nx=1E6, L=20.0), 2.26528666506213),
        ]

        for params, expected in test_data:
            actual = get_epsilon_S(**params)
            self.assertAlmostEqual(expected, actual)


    def test_get_epsilon_S_exceptions(self):
        """ Tests that get_epsilon_R raises errors when encountering instabilities."""
        with self.assertRaises(ValueError):
            get_epsilon_S(target_delta=1e-6, sigma=.5, q=0.01, ncomp=1E4, nx=1E6, L=5.0)
            get_epsilon_S(target_delta=1e-4, sigma=0.001, q=0.2, ncomp=1E4, nx=1E6, L=40.0)

if __name__ == '__main__':
    unittest.main()

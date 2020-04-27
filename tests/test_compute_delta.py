import unittest

from fourier_accountant import get_delta_R, get_delta_S

class compute_delta_regression_tests(unittest.TestCase):

    def test_get_delta_R_regression_valid_params(self):
        test_data = [
            (dict(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0), 0.010224911209191894),
            (dict(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=40.0), 0.010224911126651472),
            (dict(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E5,L=20.0), 0.010224761247217004),
            (dict(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E5,nx=1E6,L=20.0), 0.39205728853284305),
            (dict(target_eps=0.8,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0), 0.022850339595771204),
            (dict(target_eps=1.0,sigma=1.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0), 0.2391763947364977),
            (dict(target_eps=1.0,sigma=2.0,q=0.02,ncomp=1E4,nx=1E6,L=20.0), 0.14991732969314892),
        ]


        for params, expected in test_data:
            actual = get_delta_R(**params)
            self.assertAlmostEqual(expected, actual)

    def test_get_delta_S_regression_valid_params(self):
        test_data = [
            (dict(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0), 0.1282093857106282),
            (dict(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=40.0), 0.12820938556632053),
            (dict(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E5,L=20.0), 0.1282093810636456),
            (dict(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E5,nx=1E6,L=20.0), 0.8205538014405241),
            (dict(target_eps=0.8,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0), 0.16803207353601804),
            (dict(target_eps=1.0,sigma=1.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0), 0.556809673151891),
            (dict(target_eps=1.0,sigma=2.0,q=0.02,ncomp=1E4,nx=1E6,L=20.0), 0.5113118821506575),
        ]

        for params, expected in test_data:
            actual = get_delta_S(**params)
            self.assertAlmostEqual(expected, actual)


    def test_get_delta_S_instability_exceptions(self):
        with self.assertRaises(ValueError):
            get_delta_S(target_eps=1.0, sigma=.1, q=0.1, ncomp=1E4, nx=1E6, L=5.0) 


if __name__ == '__main__':
    unittest.main()

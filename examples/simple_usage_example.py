
'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

Simple example of how the functions in the package are used

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''

import fourier_accountant

ncomp = 1000  # number of compositions of DP queries over minibatches
q     = 0.01  # subsampling ratio of minibatch
sigma = 4.0   # noise level for each query

# computing delta for given epsilon for remove/add neighbouring relation
delta = fourier_accountant.get_delta_R(target_eps=1.0, sigma=sigma, q=q, ncomp=ncomp)
print(delta)
# 4.243484012034273e-06

# computing epsilon for given delta for substitute neighbouring relation
eps = fourier_accountant.get_epsilon_S(target_delta=1e-5, sigma=sigma, q=q, ncomp=ncomp)
print(eps)
# 1.9931200626285734

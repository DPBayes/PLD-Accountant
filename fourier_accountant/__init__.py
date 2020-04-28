
'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''

from .compute_eps import get_epsilon_R, get_epsilon_S
from .compute_delta import get_delta_R, get_delta_S

__all__ = [
    "get_epsilon_R",
    "get_epsilon_S",
    "get_delta_R",
    "get_delta_S"
]

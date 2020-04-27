
'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

This module holds functions for computing delta given epsilon.

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''

import numpy as np
from .common import _evaluate_pld

__all__ = ['get_delta_R', 'get_delta_S']

def _get_delta(
        relation: str, target_eps: float, sigma: float, q: float, ncomp: int, nx: int, L: float
    ):
    """
    _INTERNAL_ Computes DP delta for substite or remove/add relation.

    Internal implementation, use `get_delta_R` or `get_delta_S` instead.

    The computed delta privacy value is for the composition of ncomp subsequent
    operations over batches Poisson-subsampled with rate q from the dataset, each
    perturbed by privacy noise sigma.

    Note that this function relies on numerical approximations, which are influenced
    by choice of parameters nx and L. Increasing L roughly increases the range over
    which the integral of the privacy loss distribution is approximated, while nx is
    the number of evaluation points in [-L,L]. If you find results output by this function
    to be inaccurate, try adjusting these parameters. Refer to [1] for more details.

    Parameters:
        relation (str): Which relation to consider: _R_emove/add or _S_ubstitute
        target_eps (float): Target epsilon
        sigma (float): Privacy noise sigma
        q (float): Subsampling ratio, i.e., how large are batches relative to the dataset
        ncomp (int): Number of compositions, i.e., how many subsequent batch operations are queried
        nx (int): Number of discretiation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): delta value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    if target_eps <= 0:
        raise ValueError("target_eps must be a positive number")

    nx = int(nx)
    ncomp = int(ncomp)

    x, cfx, dx = _evaluate_pld(relation, sigma, q, ncomp, nx, L)

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+np.real(target_eps))/(2*L))))

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int = np.sum(integrand)
    delta = sum_int*dx

    if np.isnan(delta):
        raise ValueError("Computation reached a NaN value. "\
            "This can happen if sigma is chosen too small, please check the parameters.")

    return np.real(delta)


def get_delta_R(
        target_eps: float = 1.0,
        sigma: float = 2.0,
        q: float = 0.01,
        ncomp: int = int(1E4),
        nx: int = int(1E6),
        L: float = 20.0
    ):
    """
    Computes the DP delta for the remove/add neighbouring relation of datasets.

    The computed delta privacy value is for the composition of ncomp subsequent
    operations over batches Poisson-subsampled with rate q from the dataset, each
    perturbed by privacy noise sigma.

    Note that this function relies on numerical approximations, which are influenced
    by choice of parameters nx and L. Increasing L roughly increases the range over
    which the integral of the privacy loss distribution is approximated, while nx is
    the number of evaluation points in [-L,L]. If you find results output by this function
    to be inaccurate, try adjusting these parameters. Refer to [1] for more details.

    Parameters:
        target_eps (float): Target epsilon
        sigma (float): Privacy noise sigma
        q (float): Subsampling ratio, i.e., how large are batches relative to the dataset
        ncomp (int): Number of compositions, i.e., how many subsequent batch operations are queried
        nx (int): Number of discretiation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): delta value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """

    return _get_delta('R', target_eps, sigma, q, ncomp, nx, L)


def get_delta_S(
        target_eps: float = 1.0,
        sigma: float = 2.0,
        q: float = 0.01,
        ncomp: int = int(1E4),
        nx: int = int(1E6),
        L: float = 20.0
    ):
    """
    Computes the DP delta for the substitute neighbouring relation of datasets.

    The computed delta privacy value is for the composition of ncomp subsequent
    operations over batches Poisson-subsampled with rate q from the dataset, each
    perturbed by privacy noise sigma.

    Note that this function relies on numerical approximations, which are influenced
    by choice of parameters nx and L. Increasing L roughly increases the range over
    which the integral of the privacy loss distribution is approximated, while nx is
    the number of evaluation points in [-L,L]. If you find results output by this function
    to be inaccurate, try adjusting these parameters. Refer to [1] for more details.

    Parameters:
        target_eps (float): Target epsilon
        sigma (float): Privacy noise sigma
        q (float): Subsampling ratio, i.e., how large are batches relative to the dataset
        ncomp (int): Number of compositions, i.e., how many subsequent batch operations are queried
        nx (int): Number of discretiation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): delta value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    return _get_delta('S', target_eps, sigma, q, ncomp, nx, L)

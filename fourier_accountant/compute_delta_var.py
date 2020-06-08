'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

This module holds functions for computing delta given epsilon for variable sigma and q.

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''

import numpy as np
from .common_var import _evaluate_pld

__all__ = ['get_delta_R', 'get_delta_S']
def _get_delta(
        relation: str,
        sigma_t: np.ndarray,
        q_t: np.ndarray,
        k: np.ndarray,
        target_eps: float,
        nx: int,
        L: float
    ):
    """
    _INTERNAL_ Computes DP delta for substite or remove/add relation.

    Internal implementation, use `get_delta_R` or `get_delta_S` instead.

    The computed delta privacy value is for the composition of DP operations
    as specified by `sigma_t`, `q_t` and `k`, where `sigma_t` and `q_t` specify
    privacy noise and subsampling ratio for each operation and `k` is the number
    of repetitions, i.e.,
    - `k[0]` operations with privacy noise `sigma_t[0]` and subsampling ratio `q_t[0]`
    - `k[1]` operations with privacy noise `sigma_t[1]` and subsampling ratio `q_t[1]`
    - etc
    for a total of `np.sum(k)` operations.

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

    x, cfx, dx = _evaluate_pld(relation, sigma_t, q_t, k, nx, L)

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
        sigma_t: np.ndarray,
        q_t: np.ndarray,
        k: np.ndarray,
        target_eps: float = 1.0,
        nx: int = int(1E6),
        L: float = 20.0
    ):
    """
    Computes the DP delta for the remove/add neighbouring relation of datasets.

    The computed delta privacy value is for the composition of DP operations
    as specified by `sigma_t`, `q_t` and `k`, where `sigma_t` and `q_t` specify
    privacy noise and subsampling ratio for each operation and `k` is the number
    of repetitions, i.e.,
    - `k[0]` operations with privacy noise `sigma_t[0]` and subsampling ratio `q_t[0]`
    - `k[1]` operations with privacy noise `sigma_t[1]` and subsampling ratio `q_t[1]`
    - etc
    for a total of `np.sum(k)` operations.

    Note that this function relies on numerical approximations, which are influenced
    by choice of parameters nx and L. Increasing L roughly increases the range over
    which the integral of the privacy loss distribution is approximated, while nx is
    the number of evaluation points in [-L,L]. If you find results output by this function
    to be inaccurate, try adjusting these parameters. Refer to [1] for more details.

    Parameters:
        sigma_t (np.ndarray(float)): Privacy noise sigma for composed DP operations
        q_t (np.ndarray(float)): Subsampling ratios, i.e., how large are batches relative to the dataset
        k (np.ndarray(int)): Repetitions for each values in `sigma_t` and `q_t`
        target_eps (float): Target epsilon
        nx (int): Number of discretisation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): delta value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    return _get_delta('R', sigma_t, q_t, k, target_eps, nx, L)


def get_delta_S(
        sigma_t: np.ndarray,
        q_t: np.ndarray,
        k: np.ndarray,
        target_eps: float = 1.0,
        nx: int = int(1E6),
        L: float = 20.0
    ):
    """
    Computes the DP delta for the substitute neighbouring relation of datasets.

    The computed delta privacy value is for the composition of DP operations
    as specified by `sigma_t`, `q_t` and `k`, where `sigma_t` and `q_t` specify
    privacy noise and subsampling ratio for each operation and `k` is the number
    of repetitions, i.e.,
    - `k[0]` operations with privacy noise `sigma_t[0]` and subsampling ratio `q_t[0]`
    - `k[1]` operations with privacy noise `sigma_t[1]` and subsampling ratio `q_t[1]`
    - etc
    for a total of `np.sum(k)` operations.

    Note that this function relies on numerical approximations, which are influenced
    by choice of parameters nx and L. Increasing L roughly increases the range over
    which the integral of the privacy loss distribution is approximated, while nx is
    the number of evaluation points in [-L,L]. If you find results output by this function
    to be inaccurate, try adjusting these parameters. Refer to [1] for more details.

    Parameters:
        sigma_t (np.ndarray(float)): Privacy noise sigma for composed DP operations
        q_t (np.ndarray(float)): Subsampling ratios, i.e., how large are batches relative to the dataset
        k (np.ndarray(int)): Repetitions for each values in `sigma_t` and `q_t`
        target_eps (float): Target epsilon
        nx (int): Number of discretisation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): delta value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    return _get_delta('S', sigma_t, q_t, k, target_eps, nx, L)

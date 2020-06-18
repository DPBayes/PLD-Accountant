'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

This module holds functions for computing epsilon given delta for variable sigma and q.
It remains merely for compatibility to older code. Use compute_eps.get_epsilon_[S|R] instead.

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''

import numpy as np
import warnings
from .compute_eps import get_epsilon_R as get_epsilon_R_new, get_epsilon_S as get_epsilon_S_new

__all__ = ['get_epsilon_R', 'get_epsilon_S']

def get_epsilon_R(
        sigma_t: np.ndarray,
        q_t: np.ndarray,
        k: np.ndarray,
        target_delta: float = 1e-6,
        nx: int = int(1E6),
        L: float = 20.0
    ):
    """
    Computes the DP epsilon for the remove/add neighbouring relation of datasets.

    The computed epsilon privacy value is for the composition of DP operations
    as specified by `sigma_t`, `q_t` and `k`, where `sigma_t` and `q_t` specify
    privacy noise and subsampling ratio for each operation and `k` is the number
    of repetitions, i.e.,
    - `k[0]` operations with privacy noise `sigma_t[0]` and subsampling ratio `q_t[0]`
    - `k[1]` operations with privacy noise `sigma_t[1]` and subsampling ratio `q_t[1]`
    - etc
    for a total of `np.sum(k)` operations.

    Note that this function relies on numerical approximations, which are influenced
    by choice of parameters nx and L. Increasing L roughly increases the range over
    which the integral of the privacy loss distribution is approximated. L must be chosen
    large enough to cover the computed epsilon, otherwise a ValueError is raised. Try
    increasing L if this happens.

    nx is the number of evaluation points in [-L,L]. If you find results output by this
    function to be inaccurate, try adjusting these parameters. Refer to [1] for more details.
    Due to numerical instabilities, corner cases exist where this function sometimes returns
    inaccurate values. If you think this is occuring, increasing nx and verifying that
    the returned value does not change by much is usually a good heuristic to verify the output.

    Parameters:
        sigma_t (np.ndarray(float)): Privacy noise sigma for composed DP operations
        q_t (np.ndarray(float)): Subsampling ratios, i.e., how large are batches relative to the dataset
        k (np.ndarray(int)): Repetitions for each values in `sigma_t` and `q_t`
        target_delta (float): Target delta
        nx (int): Number of discretisation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): epsilon value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    warnings.warn("DEPRECATED FUNCTION! Use fourier_accountant.get_epsilon_R instead.", np.VisibleDeprecationWarning)
    return get_epsilon_R_new(target_delta, sigma_t, q_t, k, nx, L)

def get_epsilon_S(
        sigma_t: np.ndarray,
        q_t: np.ndarray,
        k: np.ndarray,
        target_delta: float = 1e-6,
        nx: int = int(1E6),
        L: float = 20.0
    ):
    """
    Computes the DP epsilon for the substitute neighbouring relation of datasets.

    The computed epsilon privacy value is for the composition of DP operations
    as specified by `sigma_t`, `q_t` and `k`, where `sigma_t` and `q_t` specify
    privacy noise and subsampling ratio for each operation and `k` is the number
    of repetitions, i.e.,
    - `k[0]` operations with privacy noise `sigma_t[0]` and subsampling ratio `q_t[0]`
    - `k[1]` operations with privacy noise `sigma_t[1]` and subsampling ratio `q_t[1]`
    - etc
    for a total of `np.sum(k)` operations.

    Note that this function relies on numerical approximations, which are influenced
    by choice of parameters nx and L. Increasing L roughly increases the range over
    which the integral of the privacy loss distribution is approximated. L must be chosen
    large enough to cover the computed epsilon, otherwise a ValueError is raised. Try
    increasing L if this happens.

    nx is the number of evaluation points in [-L,L]. If you find results output by this
    function to be inaccurate, try adjusting these parameters. Refer to [1] for more details.
    Due to numerical instabilities, corner cases exist where this function sometimes returns
    inaccurate values. If you think this is occuring, increasing nx and verifying that
    the returned value does not change by much is usually a good heuristic to verify the output.

    Parameters:
        sigma_t (np.ndarray(float)): Privacy noise sigma for composed DP operations
        q_t (np.ndarray(float)): Subsampling ratios, i.e., how large are batches relative to the dataset
        k (np.ndarray(int)): Repetitions for each values in `sigma_t` and `q_t`
        target_delta (float): Target delta
        nx (int): Number of discretisation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): epsilon value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    warnings.warn("DEPRECATED FUNCTION! Use fourier_accountant.get_epsilon_S instead.", np.VisibleDeprecationWarning)
    return get_epsilon_S_new(target_delta, sigma_t, q_t, k, nx, L)

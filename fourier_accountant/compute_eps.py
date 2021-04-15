'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

This module holds functions for computing epsilon given delta for variable sigma and q.

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''

import numpy as np
from typing import Union
from fourier_accountant.common import _evaluate_pld

__all__ = ['get_epsilon_R', 'get_epsilon_S']

def _get_epsilon(
        relation: str,
        target_delta: float,
        sigma: Union[np.ndarray, float],
        q: Union[np.ndarray, float],
        ncomp: Union[np.ndarray, int],
        nx: int,
        L: float
    ):
    """
    _INTERNAL_ Computes DP epsilon for substite or remove/add relation.

    Internal implementation, use `get_epsilon_R` or `get_epsilon_S` instead.

    The computed epsilon privacy value is for the composition of DP operations
    as specified by `sigma`, `q` and `ncomp`, where `sigma` and `q` specify
    privacy noise and subsampling ratio for each operation and `ncomp` gives the number
    of repetitions, i.e.,
    - `ncomp[0]` operations with privacy noise `sigma[0]` and subsampling ratio `q[0]`
    - `ncomp[1]` operations with privacy noise `sigma[1]` and subsampling ratio `q[1]`
    - etc
    for a total of `np.sum(ncomp)` operations. `sigma`, `q` and `ncomp` can be provided
    as scalar values if privacy noise and subsampling ratio are constant over all
    subsequent operations.

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
        relation (str): Which relation to consider: _R_emove/add or _S_ubstitute
        target_delta (float): Target delta
        sigma (np.ndarray | float): Privacy noise sigma for composed DP operations
        q (np.ndarray | float): Subsampling ratios, i.e., how large are batches relative to the dataset
        ncomp (np.ndarray | int): Repetitions for each values in `sigma` and `q`
        nx (int): Number of discretisation points
        L (float):  Limit for the approximation of the privacy loss distribution integral
    Returns:
        (float): epsilon value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    if target_delta < 0:
        raise ValueError("target_delta must be a positive number")
    if target_delta > 1:
        raise ValueError("target_delta must not exceed 1")

    nx = int(nx)

    if not isinstance(sigma, np.ndarray):
        sigma = np.array([sigma])
    if not isinstance(q, np.ndarray):
        q = np.array([q])
    if not isinstance(ncomp, np.ndarray):
        ncomp = np.array([int(ncomp)])


    x, cfx, dx = _evaluate_pld(relation, sigma, q, ncomp, nx, L)

    #Initial value \epsilon_0
    eps_0 = 0

    tol_newton = 1e-10 # set this to, e.g., 0.01*target_delta
    while True: # newton iteration to find epsilon for target_delta

        # Find first jj for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        jj = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))

        # Numerical integrands and integral domain
        dexp_e = -np.exp(eps_0-x[jj+1:])
        exp_e = 1+dexp_e

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e*cfx[jj+1:]
        integrand2 = dexp_e*cfx[jj+1:]
        sum_int = np.sum(integrand)
        sum_int2 = np.sum(integrand2)
        delta_temp = sum_int*dx
        derivative = sum_int2*dx

        if np.isnan(delta_temp):
            raise ValueError("Computation reached a NaN value. "\
                "This can happen if sigma is chosen too small, please check the parameters.")

        # Here tol is the stopping criterion for Newton's iteration
        # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)
        if np.abs(delta_temp - target_delta) <= tol_newton:
            break

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta)/derivative
        if eps_0 < -L or eps_0 > L:
            break

    eps_0 = np.real(eps_0)
    if eps_0 < -L or eps_0 > L:
        raise ValueError("Epsilon out of [-L,L] window, please check the parameters.")

    return eps_0

def get_epsilon_R(
        target_delta: float = 1e-6,
        sigma: Union[np.ndarray, float] = np.array([2.]),
        q: Union[np.ndarray, float] = np.array([0.01]),
        ncomp: Union[np.ndarray, int] = np.array([int(1E4)]),
        nx: int = int(1E6),
        L: float = 20.0
    ):
    """
    Computes the DP epsilon for the remove/add neighbouring relation of datasets.

    The computed epsilon privacy value is for the composition of DP operations
    as specified by `sigma`, `q` and `ncomp`, where `sigma` and `q` specify
    privacy noise and subsampling ratio for each operation and `ncomp` gives the number
    of repetitions, i.e.,
    - `ncomp[0]` operations with privacy noise `sigma[0]` and subsampling ratio `q[0]`
    - `ncomp[1]` operations with privacy noise `sigma[1]` and subsampling ratio `q[1]`
    - etc
    for a total of `np.sum(ncomp)` operations. `sigma`, `q` and `ncomp` can be provided
    as scalar values if privacy noise and subsampling ratio are constant over all
    subsequent operations.

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
        target_delta (float): Target delta
        sigma (np.ndarray | float): Privacy noise sigma for composed DP operations
        q (np.ndarray | float): Subsampling ratios, i.e., how large are batches relative to the dataset
        ncomp (np.ndarray | int): Repetitions for each values in `sigma` and `q`
        nx (int): Number of discretisation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): epsilon value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    return _get_epsilon('R', target_delta, sigma, q, ncomp, nx, L)

def get_epsilon_S(
        target_delta: float = 1e-6,
        sigma: Union[np.ndarray, float] = np.array([2.]),
        q: Union[np.ndarray, float] = np.array([0.01]),
        ncomp: Union[np.ndarray, int] = np.array([int(1E4)]),
        nx: int = int(1E6),
        L: float = 20.0
    ):
    """
    Computes the DP epsilon for the substitute neighbouring relation of datasets.

    The computed epsilon privacy value is for the composition of DP operations
    as specified by `sigma`, `q` and `ncomp`, where `sigma` and `q` specify
    privacy noise and subsampling ratio for each operation and `ncomp` gives the number
    of repetitions, i.e.,
    - `ncomp[0]` operations with privacy noise `sigma[0]` and subsampling ratio `q[0]`
    - `ncomp[1]` operations with privacy noise `sigma[1]` and subsampling ratio `q[1]`
    - etc
    for a total of `np.sum(ncomp)` operations. `sigma`, `q` and `ncomp` can be provided
    as scalar values if privacy noise and subsampling ratio are constant over all
    subsequent operations.

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
        target_delta (float): Target delta
        sigma (np.ndarray | float): Privacy noise sigma for composed DP operations
        q (np.ndarray | float): Subsampling ratios, i.e., how large are batches relative to the dataset
        ncomp (np.ndarray | int): Repetitions for each values in `sigma` and `q`
        nx (int): Number of discretisation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): epsilon value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    return _get_epsilon('S', target_delta, sigma, q, ncomp, nx, L)

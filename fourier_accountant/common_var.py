'''
Fourier Accountant
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.

This module holds common code to evaluate the privacy loss distribution for all computations.

The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)

The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''

import numpy as np

def _check_args(
    sigma_t: np.ndarray,
    q_t: np.ndarray,
    k: np.ndarray,
    nx: int,
    L: float
):
    if np.any(sigma_t <= 0):
        raise ValueError("all sigma must be positive numbers")
    if np.any(q_t <= 0):
        raise ValueError("all q must be positive numbers")
    if np.any(q_t > 1):
        raise ValueError("no q may exceed 1")
    if (not np.issubdtype(k.dtype, np.integer)):
        raise ValueError("all k must be of integer dtype")
    if np.any(k <= 0):
        raise ValueError("all k must be positive whole numbers")
    if nx <= 0:
        raise ValueError("nx must be a positive whole number")
    if L <= 0:
        raise ValueError("L must be a positive number")

def _evaluate_pld(
    relation: str,
    sigma_t: np.ndarray,
    q_t: np.ndarray,
    k: np.ndarray,
    nx: int,
    L: float
):
    """
    _INTERNAL_ Evaluates the privacy loss distribution, which is crucial
    for computation of both epsilon and delta.

    The PLD is evaluated for the composition of DP operations
    as specified by `sigma_t`, `q_t` and `k`, where `sigma_t` and `q_t` specify
    privacy noise and subsampling ratio for each operation and `k` is the number
    of repetitions, i.e.,
    - `k[0]` operations with privacy noise `sigma_t[0]` and subsampling ratio `q_t[0]`
    - `k[1]` operations with privacy noise `sigma_t[1]` and subsampling ratio `q_t[1]`
    - etc
    for a total of `np.sum(k)` operations.

    Parameters:
        relation (str): Which relation to consider: _R_emove/add or _S_ubstitute
        sigma_t (np.ndarray(float)): Privacy noise sigma for composed DP operations
        q_t (np.ndarray(float)): Subsampling ratios, i.e., how large are batches relative to the dataset
        k (np.ndarray(int)): Repetitions for each values in `sigma_t` and `q_t`
        target_delta (float): Target delta
        nx (int): Number of discretisation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        tuple (x, cfx, dx):
            x (np.array(float)): discretisation points for the integral of the
                                 privacy loss distribution
            cfx (np.array(float)): evaluation values
            dx (float): discretisation step length

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    _check_args(sigma_t, q_t, k, nx, L)
    assert(relation in ('S', 'R')) # assertion because this argument should only be used internally

    dx = 2.0 * L/nx # discretisation interval \Delta x
    x = np.linspace(-L, L-dx, nx, dtype=np.complex128) # grid for the numerical integration

    F_prod = np.ones(x.size)

    num_episoded = sigma_t.size

    if(q_t.size != num_episoded) or (k.size != num_episoded):
        raise ValueError('Arrays provided for sigma_t, q_t and k must all be of the same size')

    for ij in range(num_episoded):

        sigma = sigma_t[ij]
        q = q_t[ij]
        ncomp = int(k[ij])

        # Evaluate the PLD distribution,
        if relation == 'R':
            # The case of remove/add relation (Subsection 5.1)

            # first ii for which x(ii)>log(1-q),
            # i.e. start of the integral domain
            ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))

            y = x[ii+1:]
            ey = np.exp(y)
            Linvx = (sigma**2) * np.log((ey - (1-q)) / q) + 0.5
            ALinvx = (1/np.sqrt(2*np.pi*sigma**2)) * (
                    (1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
                    q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2))
                )
            dLinvx = (sigma**2) / (1 - np.exp(np.log1p(-q) - y))

            fx = np.zeros(nx)
            fx[ii+1:] = np.real(ALinvx*dLinvx)

        elif relation == 'S':
            # This is the case of substitution relation (subsection 5.2)
            c = q*np.exp(-1/(2*sigma**2))
            ey = np.exp(x)
            term1 = (-(1-q)*(1-ey) +  np.sqrt((1-q)**2*(1-ey)**2 + 4*c**2*ey))/(2*c)
            term1 = np.maximum(term1, 1e-16)
            Linvx = (sigma**2)*np.log(term1)

            sq = np.sqrt((1-q)**2*(1-ey)**2 + 4*c**2*ey)
            nom1 = 4*c**2*ey - 2*(1-q)**2*ey*(1-ey)
            term1 = nom1/(2*sq)
            nom2 = term1 + (1-q)*ey
            nom2 = nom2*(sq+(1-q)*(1-ey))
            dLinvx = sigma**2*nom2/(4*c**2*ey)

            ALinvx = (1/np.sqrt(2*np.pi*sigma**2)) * (
                    (1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
                    q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2))
                )

            fx = np.real(ALinvx*dLinvx)

        nx_half = nx // 2

        # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
        temp = np.copy(fx[nx_half:])
        fx[nx_half:] = np.copy(fx[:nx_half])
        fx[:nx_half] = temp

        FF1 = np.fft.fft(fx * dx) # Compute the DFFT
        F_prod = F_prod * FF1**ncomp
        if np.any(np.isinf(F_prod)):
            raise ValueError("Computation reached an infinite value. This can happen if sigma is "\
                "chosen too small, please check the parameters.")


    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[nx_half:])
    cfx[nx_half:] = cfx[:nx_half]
    cfx[:nx_half] = temp

    return x, cfx, dx

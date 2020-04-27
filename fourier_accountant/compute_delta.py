
'''
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.
The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)
The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko)
'''

import numpy as np

__all__ = ['get_delta', 'get_delta_S']


def check_args(target_eps, sigma, q, ncomp, nx, L):
    if target_eps <= 0:
        raise ValueError("target_eps must be a positive number")
    if sigma <= 0:
        raise ValueError("sigma must be a positive number")
    if q <= 0:
        raise ValueError("q must be a positive number")
    if q > 1:
        raise ValueError("q must not exceed 1")
    if ncomp <= 0:
        raise ValueError("ncomp must be a positive whole number")
    if nx <= 0:
        raise ValueError("nx must be a positive whole number")
    if L <=0:
        raise ValueError("L must be a positive number")


def _get_delta(relation, target_eps, sigma, q, ncomp, nx, L):
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
        Antti Koskela, Joonas Jälkö, Antti Honkela: Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049  
    """
    assert(relation == 'S' or relation =='R')
    check_args(target_eps, sigma, q, ncomp, nx, L)

    nx = int(nx)

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    # Evaluate the PLD distribution,
    if relation == 'R':
        # The case of remove/add relation (Subsection 5.1)

        # first ii for which x(ii)>log(1-q),
        # i.e. start of the integral domain
        ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))

        Linvx = (sigma**2)*np.log((np.exp(x[ii+1:])-(1-q))/q) + 0.5
        ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
            q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
        dLinvx = (sigma**2*np.exp(x[ii+1:]))/(np.exp(x[ii+1:])-(1-q));

        fx = np.zeros(nx)
        fx[ii+1:] =  np.real(ALinvx*dLinvx)
    elif relation == 'S':
        # This is the case of substitution relation (subsection 5.2)
        c = q*np.exp(-1/(2*sigma**2))
        ey = np.exp(x)
        term1=(-(1-q)*(1-ey) +  np.sqrt((1-q)**2*(1-ey)**2 + 4*c**2*ey))/(2*c)
        term1=np.maximum(term1,1e-16)
        Linvx = (sigma**2)*np.log(term1)

        sq = np.sqrt((1-q)**2*(1-ey)**2 + 4*c**2*ey)
        nom1 = 4*c**2*ey - 2*(1-q)**2*ey*(1-ey)
        term1 = nom1/(2*sq)
        nom2 = term1 + (1-q)*ey
        nom2 = nom2*(sq+(1-q)*(1-ey))
        dLinvx = sigma**2*nom2/(4*c**2*ey)

        ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
        q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)))

        fx =  np.real(ALinvx*dLinvx)

    nx_half = int(nx/2)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[nx_half:])
    fx[nx_half:] = np.copy(fx[:nx_half])
    fx[:nx_half] = temp

    FF1 = np.fft.fft(fx*dx) # Compute the DFFT

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+np.real(target_eps))/(2*L))))

    FF1_transformed = FF1**ncomp
    if np.any(np.isinf(FF1_transformed)):
        raise ValueError("Computation reached an infinite value. This can happen if sigma is chosen too small, please check the parameters.")

    FF1_transformed /= dx

    # Compute the inverse DFT
    cfx = np.fft.ifft(FF1_transformed)

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[nx_half:])
    cfx[nx_half:] = cfx[:nx_half]
    cfx[:nx_half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int*dx

    if np.isnan(delta):
        raise ValueError("Computation reached a NaN value. This can happen if sigma is chosen too small, please check the parameters.")

    return np.real(delta)


def get_delta_R(target_eps=1.0, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0):
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
        Antti Koskela, Joonas Jälkö, Antti Honkela: Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049  
    """

    return _get_delta('R', target_eps, sigma, q, ncomp, nx, L)


def get_delta_S(target_eps=1.0, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0):
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
        Antti Koskela, Joonas Jälkö, Antti Honkela: Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049  
    """
    return _get_delta('S', target_eps, sigma, q, ncomp, nx, L)
    
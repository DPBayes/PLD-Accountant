
'''
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.
The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)
The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko)
'''

import numpy as np

__all__ = ['get_epsilon_R', 'get_epsilon_S']


def check_args(target_delta, sigma, q, ncomp, nx, L):
    if target_delta < 0:
        raise ValueError("target_delta must be a positive number")
    if target_delta > 1:
        raise ValueError("target_delta must not exceed 1")
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


def _compute_eps(relation, target_delta, sigma, q, ncomp, nx, L):
    """
    _INTERNAL_ Computes DP epsilon for substite or remove/add relation.

    Internal implementation, use `get_epsilon_R` or `get_epsilon_S` instead.
    
    The computed epsilon privacy value is for the composition of ncomp subsequent
    operations over batches Poisson-subsampled with rate q from the dataset, each
    perturbed by privacy noise sigma.

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
        sigma (float): Privacy noise sigma
        q (float): Subsampling ratio, i.e., how large are batches relative to the dataset
        ncomp (int): Number of compositions, i.e., how many subsequent batch operations are queried
        nx (int): Number of discretiation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): epsilon value 

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela: Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049  
    """
    assert(relation == 'S' or relation == 'R')

    check_args(target_delta, sigma, q, ncomp, nx, L)

    nx = int(nx)

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    # Evaluate the PLD distribution
    if relation == 'R':
        # The case of remove/add relation (Subsection 5.1)

        # first ii for which x(ii+1)>log(1-q),
        # i.e. start of the integral domain
        ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))

        ey = np.exp(x[ii+1:])
        Linvx = (sigma**2)*np.log((np.exp(x[ii+1:])-(1-q))/q) + 0.5

        ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
            q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
        dLinvx = (sigma**2)*ey/(ey-(1-q));

        fx = np.zeros(nx)
        fx[ii+1:] =  np.real(ALinvx*dLinvx)
    else:
        # This is the case of substitution relation (subsection 5.2)
        ii = 1
        c = q*np.exp(-1/(2*sigma**2))
        ey = np.exp(x[ii-1:])
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
        fx = np.zeros(nx)
        fx[ii-1:] =  np.real(ALinvx*dLinvx)

    nx_half = int(nx/2)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[nx_half:])
    fx[nx_half:] = np.copy(fx[:nx_half])
    fx[:nx_half] = temp

    FF1 = np.fft.fft(fx*dx) # Compute the DFFT

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

    #Initial value \epsilon_0
    eps_0 = 0
    tol_newton = 1e-10 if relation == 'S' else 1e-13 # set this to small enough, e.g., 0.01*target_delta
    while True: # newton iteration to find epsilon for target delta

        # Find first kk for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        kk = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))

        # Numerical integrands and integral domain
        dexp_e = -np.exp(eps_0-x[kk+1:])
        exp_e = 1+dexp_e

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e*cfx[kk+1:]
        integrand2 = dexp_e*cfx[kk+1:]
        sum_int = np.sum(integrand)
        sum_int2 = np.sum(integrand2)
        delta_temp = sum_int*dx
        derivative = sum_int2*dx

        if np.isnan(delta_temp):
            raise ValueError("Computation reached a NaN value. This can happen if sigma is chosen too small, please check the parameters.")

        # Here tol is the stopping criterion for Newton's iteration
        # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)
        if np.abs(delta_temp - target_delta) <= tol_newton:
            break

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta)/derivative

        if(eps_0<-L or eps_0>L):
            break

    if(np.real(eps_0) < -L or np.real(eps_0) > L):
        raise ValueError("Epsilon out of [-L,L] window, please check the parameters.")
    else:
        return np.real(eps_0)


def get_epsilon_R(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0):
    """
    Computes the DP epsilon for the remove/add neighbouring relation of datasets.
    
    The computed epsilon privacy value is for the composition of ncomp subsequent
    operations over batches Poisson-subsampled with rate q from the dataset, each
    perturbed by privacy noise sigma.

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
        sigma (float): Privacy noise sigma
        q (float): Subsampling ratio, i.e., how large are batches relative to the dataset
        ncomp (int): Number of compositions, i.e., how many subsequent batch operations are queried
        nx (int): Number of discretiation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): epsilon value 

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela: Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049  
    """
    return _compute_eps('R', target_delta, sigma, q, ncomp, nx, L)

def get_epsilon_S(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0):
    """
    Computes the DP epsilon for the substitute neighbouring relation of datasets.
    
    The computed epsilon privacy value is for the composition of ncomp subsequent
    operations over batches Poisson-subsampled with rate q from the dataset, each
    perturbed by privacy noise sigma.

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
        sigma (float): Privacy noise sigma
        q (float): Subsampling ratio, i.e., how large are batches relative to the dataset
        ncomp (int): Number of compositions, i.e., how many subsequent batch operations are queried
        nx (int): Number of discretiation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): epsilon value 

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela: Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049  
    """
    return _compute_eps('S', target_delta, sigma, q, ncomp, nx, L)

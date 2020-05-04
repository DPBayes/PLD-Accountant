





'''
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.
The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)
The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko)
'''



import numpy as np



def get_delta_R(
        sigma_t: np.ndarray,
        q_t: np.ndarray,
        k: np.ndarray,
        target_eps: float = 1.0,
        nx: int = 1E6,
        L: int = 20.0
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
        nx (int): Number of discretiation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): delta value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """

    nx = int(nx)

    tol_newton = 1e-10 # set this to, e.g., 0.01*target_delta

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    fx_table=[]
    F_prod=np.ones(x.size)

    ncomp = sigma_t.size

    if(q_t.size != ncomp) or (k.size != ncomp):
        raise ValueError('Arrays provided for sigma_t, q_t and k must all be of the same size')

    for ij in range(ncomp):

        sigma=sigma_t[ij]
        q=q_t[ij]

        # first ii for which x(ii)>log(1-q),
        # i.e. start of the integral domain
        ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))

        # Evaluate the PLD distribution,
        # The case of remove/add relation (Subsection 5.1)
        Linvx = (sigma**2)*np.log((np.exp(x[ii+1:])-(1-q))/q) + 0.5
        ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
        	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
        dLinvx = (sigma**2*np.exp(x[ii+1:]))/(np.exp(x[ii+1:])-(1-q));

        fx = np.zeros(nx)
        fx[ii+1:] =  np.real(ALinvx*dLinvx)
        half = int(nx/2)

        # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
        temp = np.copy(fx[half:])
        fx[half:] = np.copy(fx[:half])
        fx[:half] = temp

        # Compute the DFT
        FF1 = np.fft.fft(fx*dx)
        F_prod = F_prod*FF1**k[ij]

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int*dx

    return np.real(delta)





def get_delta_S(
        sigma_t: np.ndarray,
        q_t: np.ndarray,
        k: np.ndarray,
        target_eps: float = 1.0,
        nx: int = 1E6,
        L: int = 20.0
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
        nx (int): Number of discretiation points
        L (float):  Limit for the approximation of the privacy loss distribution integral

    Returns:
        (float): delta value

    References:
        Antti Koskela, Joonas Jälkö, Antti Honkela:
        Computing Tight Differential Privacy Guarantees Using FFT
            https://arxiv.org/abs/1906.03049
    """
    nx = int(nx)

    tol_newton = 1e-10 # set this to, e.g., 0.01*target_delta

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    fx_table=[]
    F_prod=np.ones(x.size)

    ncomp=sigma_t.size

    if(q_t.size != ncomp) or (k.size != ncomp):
        raise ValueError('Arrays provided for sigma_t, q_t and k must all be of the same size')

    for ij in range(ncomp):

        sigma=sigma_t[ij]
        q=q_t[ij]

        # Evaluate the PLD distribution,
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
        half = int(nx/2)

        # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
        temp = np.copy(fx[half:])
        fx[half:] = np.copy(fx[:half])
        fx[:half] = temp

        FF1 = np.fft.fft(fx*dx) # Compute the DFFT
        F_prod = F_prod*FF1**k[ij]

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+np.real(target_eps))/(2*L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int*dx

    return np.real(delta)






'''
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.
The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)
The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko)
'''



import numpy as np


def get_delta_R(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0):
    """
    This function returns the delta as a function of epsilon,
    for the case of Poisson subsampling with the remove/add neighbouring relation of datasets.

    Parameters:
      target_eps - target epsilon
      sigma - noise sigma
      q - subsampling ratio
      nx - number of points in the discretisation grid
      L -  limit for the integral
      ncomp - number of compositions
    """

    nx = int(nx)

    tol_newton = 1e-10 # set this to, e.g., 0.01*target_delta

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

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

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((FF1**ncomp/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int*dx

  
    if np.isnan(delta):
        raise ValueError("Computation reached a NaN value. This can happen if sigma is chosen too small, please check the parameters.")

    # print('DP-delta (in R-relation) after ' + str(int(ncomp)) + ' compositions:' + str(np.real(delta)) + ' (epsilon=' + str(target_eps) + ')')

    return np.real(delta)





def get_delta_S(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0):

    """
    This function returns the delta as a function of epsilon,
    for the case of Poisson subsampling with the substitute neighbouring relation of datasets.

    Parameters:
      target_eps - target epsilon
      sigma - noise sigma
      q - subsampling ratio
      nx - number of points in the discretisation grid
      L -  limit for the integral
      ncomp - number of compositions
    """


    nx = int(nx)

    tol_newton = 1e-10 # set this to, e.g., 0.01*target_delta

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

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
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int*dx


    if np.isnan(delta):
        raise ValueError("Computation reached a NaN value. This can happen if sigma is chosen too small, please check the parameters.")

    # print('DP-delta (in S-relation) after ' + str(int(ncomp)) + ' compositions:' + str(np.real(delta)) + ' (epsilon=' + str(target_eps) + ')')
    return np.real(delta)

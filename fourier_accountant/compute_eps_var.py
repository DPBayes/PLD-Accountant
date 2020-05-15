





'''
Code for computing tight DP guarantees for the subsampled Gaussian mechanism.
The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Tight Differential Privacy Guarantees Using FFT.
arXiv preprint arXiv:1906.03049 (2019)
The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko) and
was refactored by Lukas Prediger (@lumip) .
'''




import numpy as np

def get_epsilon_R(sigma_t,q_t,k,target_delta=1e-6,nx=1E6,L=20.0):

    """
    This function returns the epsilon as a function of delta,
    for the case of Poisson subsampling with the remove/add neighbouring relation of datasets.
    The function computes epsilon for varying parameters sigma and q,
    the input is given as an array of parameters.

    Parameters:
      target_eps - target epsilon
      sigma_t - array of sigmas
      q_t - array of subsampling ratios
      nx - number of points in the discretisation grid
      L -  limit for the integral
      ncomp - number of compositions
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

        # first ii for which x(ii)>log(1-q),
        # i.e. start of the integral domain
        ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))

        # Evaluate the PLD distribution,
        # The case of remove/add relation (Subsection 5.1)
        Linvx = (sigma**2)*np.log((np.exp(x[ii+1:])-(1-q))/q) + 0.5
        ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
        	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
        ey = np.exp(x[ii+1:])
        dLinvx = (sigma**2)/(1-(1-q)/ey);

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

    #Initial value \epsilon_0
    eps_0 = 0

    exp_e = 1-np.exp(eps_0-x)
    # first jj for which 1-exp(eps_0-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+eps_0)/(2*L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(eps_0) and \delta'(eps_0)
    dexp_e = -np.exp(eps_0-x[jj+1:])
    exp_e = 1+dexp_e
    integrand = exp_e*cfx[jj+1:]
    integrand2 = dexp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    sum_int2=np.sum(integrand2)
    delta_temp = sum_int*dx
    derivative = sum_int2*dx
    if np.isnan(delta_temp):
        raise ValueError("Computation reached a NaN value. "\
            "This can happen if sigma is chosen too small, please check the parameters.")

    # Here tol is the stopping criterion for Newton's iteration
    # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)
    while np.abs(delta_temp - target_delta) > tol_newton:

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta)/derivative

        if(eps_0<-L or eps_0>L):
            break

        # first kk for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        kk = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))

        # Integrands and integral domain
        dexp_e = -np.exp(eps_0-x[kk+1:])
        exp_e = 1+dexp_e

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e*cfx[kk+1:]
        integrand2 = dexp_e*cfx[kk+1:]
        sum_int=np.sum(integrand)
        sum_int2=np.sum(integrand2)
        delta_temp = sum_int*dx
        derivative = sum_int2*dx
        if np.isnan(delta_temp):
            raise ValueError("Computation reached a NaN value. "\
                "This can happen if sigma is chosen too small, please check the parameters.")

    eps_0 = np.real(eps_0)
    if eps_0 < -L or eps_0 > L:
        raise ValueError("Epsilon out of [-L,L] window, please check the parameters.")
    else:
        return eps_0






def get_epsilon_S(sigma_t,q_t,k,target_delta=1e-6,nx=1E6,L=20.0):

    """
    This function returns the epsilon as a function of delta,
    for the case of Poisson subsampling with the substitute neighbouring relation of datasets.
    The function computes epsilon for varying parameters sigma and q,
    the input is given as an array of parameters.

    Parameters:
      target_eps - target epsilon
      sigma_t - array of sigmas
      q_t - array of subsampling ratios
      nx - number of points in the discretisation grid
      L -  limit for the integral
      ncomp - number of compositions
    """


    nx = int(nx)

    tol_newton = 1e-10 # set this to, e.g., 0.01*target_delta

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    #Initial value \epsilon_0
    eps_0 = 0

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
        ey = np.exp(x)
        c = q*np.exp(-1/(2*sigma**2))
        term1=(-(1-q)*(1-ey) +  np.sqrt((1-q)**2*(1-ey)**2 + 4*c**2*ey))/(2*c)
        term1=np.maximum(term1,1e-16)
        Linvx = (sigma**2)*np.log(term1)

        sq = np.sqrt((1-q)**2*(1-ey)**2 + 4*c**2*ey)
        nom1 = 4*c**2*ey - 2*(1-q)**2*ey*(1-ey)
        term1 = nom1/(2*sq)
        nom2 = term1 + (1-q)*ey
        nom2 = nom2*(sq+(1-q)*(1-ey))
        dLinvx = sigma**2*nom2/(4*c**2*ey)


        ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2))
            + q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)))

        fx =  np.real(ALinvx*dLinvx)
        half = int(nx/2)

        # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
        temp = np.copy(fx[half:])
        fx[half:] = np.copy(fx[:half])
        fx[:half] = temp

        FF1 = np.fft.fft(fx*dx) # Compute the DFFT
        F_prod = F_prod*FF1**k[ij]

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod/dx))

    # first jj for which 1-exp(eps_0-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(eps_0) and \delta'(eps_0)
    dexp_e = -np.exp(eps_0-x[jj+1:])
    exp_e = 1+dexp_e
    integrand = exp_e*cfx[jj+1:]
    integrand2 = dexp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    sum_int2=np.sum(integrand2)
    delta_temp = sum_int*dx
    derivative = sum_int2*dx

    if np.isnan(delta_temp):
        raise ValueError("Computation reached a NaN value. "\
            "This can happen if sigma is chosen too small, please check the parameters.")

    # Here tol is the stopping criterion for Newton's iteration
    # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)
    while np.abs(delta_temp - target_delta) > tol_newton:

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta)/derivative

        if(eps_0<-L or eps_0>L):
            break

        # first kk for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        kk = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))

        # Integrands and integral domain
        dexp_e = -np.exp(eps_0-x[kk+1:])
        exp_e = 1+dexp_e

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e*cfx[kk+1:]
        integrand2 = dexp_e*cfx[kk+1:]
        sum_int=np.sum(integrand)
        sum_int2=np.sum(integrand2)
        delta_temp = sum_int*dx
        derivative = sum_int2*dx
        if np.isnan(delta_temp):
            raise ValueError("Computation reached a NaN value. "\
                "This can happen if sigma is chosen too small, please check the parameters.")


    eps_0 = np.real(eps_0)
    if eps_0 < -L or eps_0 > L:
        raise ValueError("Epsilon out of [-L,L] window, please check the parameters.")
    else:
        return eps_0
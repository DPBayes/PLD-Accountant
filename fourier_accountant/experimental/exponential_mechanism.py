"""
Experimental implementation for computing tight DP guarantees for the exponential mechanism.
The method is described in the manuscript
Tight Approximate Differential Privacy for Discrete-Valued Mechanisms Using FFT .
"""

import numpy as np

def get_delta_upper(target_eps=1.0,eps_em=0.1,ncomp=500,nx=1E6,L=20.0,m=50,n=100):
    """
    Calculates the upper bound for delta given a target epsilon for
    ncomp-fold composition of exponential mechanism with privacy value eps_em
    where the quality score is a counting query.

    Args:
        target_eps: The targeted value for epsilon of the composition.
        eps_em: The epsilon value of the mechanism under composition.
        ncomp: Number of compositions of the mechanism.
        nx: Number of points in the discretisation grid.
        L: Limit for the approximation integral.
        m: Number of elements accepted/counted by the query.
        n: Total number of elements in the counting query.
    """

    nx = int(nx)
    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    #Determine the neghbouring distributions
    P1=np.array([np.exp(eps_em*m),  np.exp(eps_em*(n-m))])
    P1=P1/np.sum(P1)

    P2=np.array([np.exp(eps_em*(m-1)),  np.exp(eps_em*(n-m))])
    P2=P2/np.sum(P2)

    Lx=np.log(P1/P2)

    #Compute the lambda-divergence \alpha^+
    lambda_sum_plus=0
    lambd=L/2
    k=ncomp
    for i in range(0,len(Lx)):
        lambda_sum_plus+=(P1[i]/P2[i])**lambd*P1[i]
    alpha_plus=np.log(lambda_sum_plus)

    #Compute the lambda-divergence \alpha^-
    lambda_sum_minus=0
    k=ncomp
    for i in range(0,len(Lx)):
        lambda_sum_minus+=(P2[i]/P1[i])**lambd*P2[i]
    alpha_minus=np.log(lambda_sum_minus)

    #Evaluate the bound of Thm. 10
    T1=(2*np.exp((ncomp+1)*alpha_plus) - np.exp((ncomp)*alpha_plus) - np.exp(alpha_plus) )/(np.exp(alpha_plus - 1))
    T2=(np.exp((ncomp+1)*alpha_minus) - np.exp(alpha_minus) )/(np.exp(alpha_minus - 1))
    error_term= (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-lambd*L)))
    print('error: ' + str(error_term))

    omega_y=np.zeros(nx)


    for i in range(0,len(Lx)):
        ii = int(np.ceil((L+Lx[i])/dx))
        omega_y[ii]+=P1[i]
        # ii = int(np.ceil((L+Lx[i])/dx))
        # omega_upper[ii]+=P1[i]
        # ii = int(np.floor((L+Lx[i])/dx))
        # omega_lower[ii]+=P1[i]

    print('Lx : ' + str(Lx))
    fx = omega_y
    half = int(nx/2)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    # Compute the DFT
    FF1 = np.fft.fft(fx)

    # Take elementwise powers and compute the inverse DFT
    cfx = np.fft.ifft((FF1**ncomp))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    sum=np.sum(cfx)

    print('Sum here: ' + str(sum)) # should be 1
    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x)
    integrand = exp_e*cfx
    sum_int=np.sum(integrand[jj+1:])
    delta = sum_int
    delta += error_term
    print('Unbounded DP-delta after ' + str(int(ncomp)) + ' compositions:' + str(np.real(delta)) + ' (epsilon=' + str(target_eps) + ')')
    return np.real(delta)





# Parameters:
# target_eps - target epsilon
# mech_eps - parameter epsilon of the mechanism
# nx - number of points in the discretisation grid
# L -  limit for the integral
# ncomp - compute for ncomp number of compositions

def get_delta_lower(target_eps=1.0,eps_em=0.1,ncomp=500,nx=1E6,L=20.0,m=50,n=100):
    """
    Calculates the lower bound for delta given a target epsilon for
    ncomp-fold composition of exponential mechanism with privacy value eps_em
    where the quality score is a counting query.

    Args:
        target_eps: The targeted value for epsilon of the composition.
        eps_em: The epsilon value of the mechanism under composition.
        ncomp: Number of compositions of the mechanism.
        nx: Number of points in the discretisation grid.
        L: Limit for the approximation integral.
        m: Number of elements accepted/counted by the query.
        n: Total number of elements in the counting query.
    """


    nx = int(nx)
    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    #Determine the neghbouring distributions
    P1=np.array([np.exp(eps_em*m),  np.exp(eps_em*(n-m))])
    P1=P1/np.sum(P1)

    P2=np.array([np.exp(eps_em*(m-1)),  np.exp(eps_em*(n-m))])
    P2=P2/np.sum(P2)

    Lx=np.log(P1/P2)

    #Compute the lambda-divergence \alpha^+
    lambda_sum_plus=0
    lambd=L/2
    k=ncomp
    for i in range(0,len(Lx)):
        lambda_sum_plus+=(P1[i]/P2[i])**lambd*P1[i]
    alpha_plus=np.log(lambda_sum_plus)

    #Compute the lambda-divergence \alpha^-
    lambda_sum_minus=0
    k=ncomp
    for i in range(0,len(Lx)):
        lambda_sum_minus+=(P2[i]/P1[i])**lambd*P2[i]
    alpha_minus=np.log(lambda_sum_minus)

    #Evaluate the bound of Thm. 10
    T1=(2*np.exp((ncomp+1)*alpha_plus) - np.exp((ncomp)*alpha_plus) - np.exp(alpha_plus) )/(np.exp(alpha_plus - 1))
    T2=(np.exp((ncomp+1)*alpha_minus) - np.exp(alpha_minus) )/(np.exp(alpha_minus - 1))
    error_term= (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-lambd*L)))

    omega_y=np.zeros(nx)


    for i in range(0,len(Lx)):
        ii = int(np.floor((L+Lx[i])/dx))
        omega_y[ii]+=P1[i]
        # ii = int(np.ceil((L+Lx[i])/dx))
        # omega_upper[ii]+=P1[i]
        # ii = int(np.floor((L+Lx[i])/dx))
        # omega_lower[ii]+=P1[i]

    fx = omega_y
    half = int(nx/2)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    # Compute the DFT
    FF1 = np.fft.fft(fx)

    # Take elementwise powers and compute the inverse DFT
    cfx = np.fft.ifft((FF1**ncomp))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    sum=np.sum(cfx)

    assert(np.allclose(sum, 1.))

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x)
    integrand = exp_e*cfx
    sum_int=np.sum(integrand[jj+1:])
    delta = sum_int
    delta -= error_term

    return np.real(delta)

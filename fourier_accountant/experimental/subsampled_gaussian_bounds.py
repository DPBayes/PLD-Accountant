"""
Experimental implementation for computing tight DP guarantees for the Gaussian mechanism.
The method is described in the manuscript
Tight Approximate Differential Privacy for Discrete-Valued Mechanisms Using FFT .
"""

import numpy as np
from scipy import optimize

def f(x, sigma, q):
    """
    Computes the value of the PLD.

    Args:
        x: Discretized evaluation points.
        sigma: Gaussian mechanism noise level.
        q: Subsampling ratio.
    """
    Linvx = (sigma**2)*np.log((np.exp(x)-(1-q))/q) + 0.5
    ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
    	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)))
    dLinvx = sigma**2*np.exp(x)/(np.exp(x)-(1-q))
    omega=ALinvx*dLinvx
    return omega

def df(x, sigma, q):
    """
    Computes the absolute value of the derivative of the PLD.

    Args:
        x: Discretized evaluation points.
        sigma: Gaussian mechanism noise level.
        q: Subsampling ratio.
    """
    Linvx = (sigma**2)*np.log((np.exp(x)-(1-q))/q) + 0.5
    ALinvx = ((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
    	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)))
    dALinvx = -2*Linvx*(1-q)*np.exp(-Linvx*Linvx/(2*sigma**2))/(2*sigma**2) - 2*(Linvx-1)*q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2))/(2*sigma**2)
    dLinvx = sigma**2*np.exp(x)/(np.exp(x)-(1-q))
    ddLinvx = -(1-q)*sigma**2*np.exp(x)/((np.exp(x)-(1-q))**2)
    domega=dALinvx*(dLinvx**2) + ALinvx*ddLinvx
    return abs(domega)

def get_delta_max(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0):
    """
    Computes upper bound for delta for a given target epsilon for compositions of
    Gaussian mechanisms with Poisson subsampling for the remove/add neighbouring
    relation of datasets.

    Args:
        target_eps: Target epsilon.
        sigma: Gaussian mechanism noise level.
        q: Subsampling ratio.
        ncomp: Number of compositions of Gaussian mechanisms.
        nx: Number of points in the discretisation grid.
        L: Limit for the approximation integral.
    """

    #Find the peak point of the PLD
    xd0 = optimize.fmin(df,x0=0,args=(sigma,q),xtol=1e-9)
    fxd0 = f(xd0,sigma,q)
    nx = int(nx)

    dx = 2.0*L/nx # discretisation interval \Delta x

    x = np.linspace(-L,L,nx+1,dtype=np.complex128) # grid for the numerical integration

    # first ii for which x(ii)>log(1-q),
    # i.e. start of the integral domain
    ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))

    # Evaluate the PLD distribution,
    # The case of remove/add relation (Subsection 5.1)
    Linvx = (sigma**2)*np.log((np.exp(x[ii+1:])-(1-q))/q) + 0.5
    ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
    	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
    dLinvx = (sigma**2*np.exp(x[ii+1:]))/(np.exp(x[ii+1:])-(1-q))
    fx_ref = np.zeros(nx+1)
    fx_ref[ii+1:] =  np.real(ALinvx*dLinvx)

    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration
    fx = np.zeros(nx)

    #Majorant for fx
    jj = int(np.ceil(float(nx*(L+xd0)/(2*L))))
    fx[:jj]=fx_ref[:jj]
    fx[jj] = fxd0
    fx[jj+1:]=fx_ref[jj:-2]

    # The 4 lines above are the vectorisation of the following:
    # for i in range(nx):
    #     if x[i]<xd0:
    #         fx[i]=fx_ref[i]
    #     elif x[i] > xd0 and x[i-1] < xd0:
    #         fx[i] = fxd0
    #     else:
    #         fx[i] = fx_ref[i-1]

    #Compute the log MGFs
    lambd=L/2
    alpha_plus=np.log(np.sum(np.exp(lambd*x)*dx*fx))
    alpha_minus=np.log(np.sum(np.exp(lambd*np.flip(x))*dx*np.flip(fx)))

    T1=(2*np.exp((ncomp+1)*alpha_plus) - np.exp((ncomp)*alpha_plus) - np.exp(alpha_plus) )/(np.exp(alpha_plus - 1))
    T2=(np.exp((ncomp+1)*alpha_minus) - np.exp(alpha_minus) )/(np.exp(alpha_minus - 1))
    error_term= (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-lambd*L)))

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
    cfx = np.fft.ifft((FF1**ncomp))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int + error_term

    return np.real(delta)



def get_delta_min(target_eps=1.0,sigma=2.0,q=0.01,ncomp=1E4,nx=1E6,L=20.0):
    """
    Computes lower bound for delta for a given target epsilon for compositions of
    Gaussian mechanisms with Poisson subsampling for the remove/add neighbouring
    relation of datasets.

    Args:
        target_eps: Target epsilon.
        sigma: Gaussian mechanism noise level.
        q: Subsampling ratio.
        ncomp: Number of compositions of Gaussian mechanisms.
        nx: Number of points in the discretisation grid.
        L: Limit for the approximation integral.
    """

    #Find the peak point of the PLD
    xd0 = optimize.fmin(df,x0=0,args=(sigma,q),xtol=1e-9)
    # fxd0 = f(xd0,sigma,q)
    nx = int(nx)

    dx = 2.0*L/nx # discretisation interval \Delta x

    x = np.linspace(-L,L,nx+1,dtype=np.complex128) # grid for the numerical integration

    # first ii for which x(ii)>log(1-q),
    # i.e. start of the integral domain
    ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))

    # Evaluate the PLD distribution,
    # The case of remove/add relation (Subsection 5.1)
    Linvx = (sigma**2)*np.log((np.exp(x[ii+1:])-(1-q))/q) + 0.5
    ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
    	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
    dLinvx = (sigma**2*np.exp(x[ii+1:]))/(np.exp(x[ii+1:])-(1-q));
    fx_ref = np.zeros(nx+1)
    fx_ref[ii+1:] =  np.real(ALinvx*dLinvx)



    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration
    fx = np.zeros(nx)

    #Minorant for fx
    jj = int(np.floor(float(nx*(L+xd0)/(2*L))))
    fx[:jj]=fx_ref[:jj]
    fx[jj] = min(f(x[jj],sigma,q),f(x[jj+1],sigma,q))
    fx[jj+1:]=fx_ref[jj+2:]

    # The 4 lines above are the vectorisation of the following:
    # for i in range(nx):
    #     if x[i]<xd0-dx:
    #         fx[i]=fx_ref[i]
    #     elif x[i] < xd0 and x[i+1] > xd0:
    #         fx[i] = min(f(x[i],sigma,q),f(x[i+1],sigma,q))
    #     else:
    #         fx[i] = fx_ref[i+1]

    #Compute the log MGFs
    lambd=L/2
    alpha_plus=np.log(np.sum(np.exp(lambd*x)*dx*fx))
    alpha_minus=np.log(np.sum(np.exp(lambd*np.flip(x))*dx*np.flip(fx)))

    T1=(2*np.exp((ncomp+1)*alpha_plus) - np.exp((ncomp)*alpha_plus) - np.exp(alpha_plus) )/(np.exp(alpha_plus - 1))
    T2=(np.exp((ncomp+1)*alpha_minus) - np.exp(alpha_minus) )/(np.exp(alpha_minus - 1))
    error_term= (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-lambd*L)))

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
    cfx = np.fft.ifft((FF1**ncomp))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int - error_term

    return np.real(delta)



import numpy as np
from scipy import optimize
import math


def get_delta_max(target_eps=1.0,sigma=2.0,ncomp=1E4,nx=1E6,L=20.0,p=0.5):

    nx = int(nx)

    dx = 2.0*L/nx # discretisation interval \Delta x

    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    fxG = np.zeros(nx)
    mu=1/(2*sigma**2)
    ss=1/(sigma**2)

    for i in range(0,int(nx/2)+1):
        fxG[i] = dx*(1/np.sqrt(2*math.pi*ss))*np.exp(-(x[i] - mu)**2/(2*ss))
    for i in range(int(nx/2)+1,nx):
        fxG[i] = dx*(1/np.sqrt(2*math.pi*ss))*np.exp(-(x[i-1] - mu)**2/(2*ss))

    # PLD distribution for the randomised response
    c_p = np.log(p/(1-p))
    i_cu = int(np.ceil((L+c_p)/dx))
    i_cd = int(np.ceil((L-c_p)/dx))
    fxR = np.zeros(nx)
    fxR[i_cu] = p
    fxR[i_cd] = 1-p

    print('fxR sum: ' + str(np.sum(fxR)))
    print('fxG sum: ' + str(np.sum(fxG)))


    #Compute the log MGFs
    lambd=L/2
    alpha_plusG=np.log(np.sum(np.exp(lambd*x)*fxG))
    alpha_plusR=np.log(np.sum(np.exp(lambd*x)*fxR))

    alpha_minusG=np.log(np.sum(np.exp(lambd*np.flip(x))*dx*np.flip(fxG)))
    alpha_minusR=np.log(np.sum(np.exp(lambd*np.flip(x))*dx*np.flip(fxR)))

    alpha_plus=max(alpha_plusG,alpha_plusR)
    alpha_minus=max(alpha_minusG,alpha_minusR)

    T1=(2*np.exp((ncomp+1)*alpha_plus) - np.exp((ncomp)*alpha_plus) - np.exp(alpha_plus) )/(np.exp(alpha_plus - 1))
    T2=(np.exp((ncomp+1)*alpha_minus) - np.exp(alpha_minus) )/(np.exp(alpha_minus - 1))
    error_term= (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-2*lambd*L)))

    print('error: ' + str(error_term))

    half = int(nx/2)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fxG[half:])
    fxG[half:] = np.copy(fxG[:half])
    fxG[:half] = temp

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fxR[half:])
    fxR[half:] = np.copy(fxR[:half])
    fxR[:half] = temp

    # Compute the DFTs
    FFG = np.fft.fft(fxG)
    FFR = np.fft.fft(fxR)
    FF1 = FFG*FFR
    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((FF1**(ncomp/2)))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int + error_term

    print('DP-delta (in R-relation) after ' + str(int(ncomp)) + ' compositions:' + str(np.real(delta)) + ' (epsilon=' + str(target_eps) + ')')

    return np.real(delta)

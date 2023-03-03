"""
Computing tight DP guarantees for the strong adversary
"""

import numpy as np
import scipy
import scipy.special
import math


#  Probabilities for the case a>0
def get_logP(n,i,j,gamma,k):

    p=gamma/k

    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(i+1) - scipy.special.gammaln(j+1) - scipy.special.gammaln(n-i-j)

    return term1 + i*np.log(p)+j*np.log(p) + (n-1-i-j)*np.log(1-2*p)



def get_omega(n,gamma,k,nx,L):


    P1 = []
    Lx = []

    p=gamma/k

    p2 = p/(1-p)
    # Throw away very small tails, using Hoeffding
    tol_ = 40

    # Bin(n-1,p)
    lower_i=int(max(0,np.floor((n)*(p-np.sqrt(tol_/(2*(n-1)))))))
    upper_i=int(min(n,np.ceil((n)*(p+np.sqrt(tol_/(2*(n-1)))))))

    print('total i: ' + str(upper_i - lower_i))

    for i in range(lower_i,upper_i):

        if i%100 == 0:
            print(i)

        lower_j=int(max(1,np.floor((n-i)*(p2-np.sqrt(tol_/(2*((n-1-i)+1)))))))
        upper_j=int(min((n-i),np.ceil((n-i)*(p2+np.sqrt(tol_/(2*((n-1-i)+1)))))))

        for j in range(lower_j,upper_j):
            p_temp=get_logP(n,i,j,gamma,k)
            P1.append(p_temp)
            Lx.append(np.log((i+1)/j))

    dx=2*L/nx
    grid_x=np.linspace(-L,L-dx,nx)
    omega_y=np.zeros(nx)
    len_lx=len(Lx)
    for i in range(0,len_lx):
        lx=Lx[i]
        if lx>-L and lx<L:
            # place the mass to the right end point of the interval -> upper bound for delta
            ii=int(np.ceil((lx+L)/dx))
            omega_y[ii]=omega_y[ii]+np.exp(P1[i])
        else:
            print('OUTSIDE OF [-L,L] - RANGE')

    # check the mass to be sure
    print('Sum of Probabilities : ' + str(sum(omega_y)))


    # Compute the periodisation & truncation error term,
    # using the error analysis given in
    # Koskela, Antti, et al.
    # Tight differential privacy for discrete-valued mechanisms and
    # for the subsampled gaussian mechanism using fft.
    # International Conference on Artificial Intelligence and Statistics. PMLR, 2021.

    # the parameter lambda in the error bound can be chosen freely.
    # commonly the error term becomes negligible for reasonable values of L.
    # here lambda is just hand tuned to make the error term small.

    lambd=L
    lambda_sum_minus=0

    for i in range(len(Lx)):
        plf = -Lx[i]
        lambda_sum_minus+=np.exp(lambd*plf + P1[i])

    alpha_minus=np.log(lambda_sum_minus)

    print(alpha_minus)

    #Then alpha plus
    lambda_sum_plus=0
    for i in range(len(Lx)):
        plf = -Lx[i]
        lambda_sum_plus+=np.exp(lambd*plf + P1[i])

    alpha_plus=np.log(lambda_sum_plus)

    #Evaluate the periodisation error bound using alpha_plus and alpha_minus
    T1=(np.exp(k*alpha_plus)  )
    T2=(np.exp(k*alpha_minus) )
    error_term = (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-2*lambd*L)))
    print('error_term: ' + str(error_term))

    return omega_y,grid_x, error_term


def get_epsilons(omega_y,error_term,grid_x,n, gamma, k, n_c, nx=int(1E6), L=1000, target_delta=1e-6):

    dx=2*L/nx

    omega_y = omega_y/np.sum(omega_y)

    half = int(nx/2)
    fx=np.copy(omega_y)

    # Compute the log MGFs and compute the error induced by the Fourier accountant,
    # using the analaysis provided by Koskela et al. (2021)
    k=n_c

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    # Compute the DFT
    FF1 = np.fft.fft(fx)
    y = np.fft.ifft((FF1**n_c))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(y[half:])
    y[half:] = y[:half]
    y[:half] = temp

    eps1=100.0
    eps0=0.0

    dd = target_delta
    sum_deltax=1.0
    tol=1e-5
    delta_eps=1.0

    epsm=(eps1+eps0)/2
    d_eps=(eps1-eps0)/2


    # Find epsilon using the bisection method
    while abs(d_eps)>tol:

        ssi=int(np.ceil((epsm+L)/dx))
        sum_delta_m=np.real(np.sum((1-np.exp(epsm-grid_x[ssi:]))*y[ssi:])) + error_term

        ssi=int(np.ceil((eps0+L)/dx))
        sum_delta_0=np.real(np.sum((1-np.exp(eps0-grid_x[ssi:]))*y[ssi:])) + error_term

        if math.copysign(1, sum_delta_m - dd) == math.copysign(1, sum_delta_0 - dd):
            eps0 = epsm
        else:
            eps1 = epsm

        epsm=(eps1+eps0)/2
        d_eps=(eps1-eps0)/2
        print(epsm)

    return eps0

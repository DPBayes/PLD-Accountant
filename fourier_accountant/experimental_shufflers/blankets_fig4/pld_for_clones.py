"""
Experimental implementation ....
"""

import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import time



#  Probabilities for the case a>0
def get_logP2(i,j,n, eps0,gamma):



    a=j+1
    b=i-j

    #mixture probability q = p exp(\veps_0)

    p_in = 1/(3+np.exp(eps0))

    q=np.exp(eps0)*p_in

    # parameter for C
    p=2*p_in

    p2=p_in/(1-np.exp(eps0)*p_in)

    # The following gives the subsampled distribution
    #    gamma P + (1-gamma) Q
    #    = gamma ( q P_1 + (1-q) P_0 ) + (1-gamma) ( (1-q) P_1 + q P_0)


    P= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
    Q= q*(b/a)+p2*(1-q)  + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
    # term0 = log(  fact1 + fact2*P(P_0=(a,b))/P(P_1=(a,b)))
    term0 = np.log(gamma*P+(1-gamma)*Q)
    # Then term0 is multiplied by P(P_1=(a,b))
    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(i+1) - scipy.special.gammaln(n-i)
    term2 = scipy.special.gammaln(i+1)-scipy.special.gammaln(j+1)-scipy.special.gammaln(i-j+1)

    return term0 + term1 + term2 + i*np.log(p)+(n-1-i)*np.log(1-p)+i*np.log(0.5)

#  Probabilities for the case a>0
def get_logP2(i,j,n, eps0,gamma):

    a=j+1
    b=i-j
    q=np.exp(eps0)/(1+np.exp(eps0))
    p=np.exp(-eps0)

    # The following gives the subsampled distribution
    #    gamma P + (1-gamma) Q
    # = gamma ( q P_1 + (1-q) P_0 ) + (1-gamma) ( (1-q) P_1 + q P_0)
    fact1 = gamma*q+(1-q)*(1-gamma)
    fact2 = gamma*(1-q) + q*(1-gamma)
    # term0 = log(  fact1 + fact2*P(P_0=(a,b))/P(P_1=(a,b)))
    term0 = np.log(fact1 + fact2*(p/(1-p))*(n-a-b)/(2*a))
    # Then term0 is multiplied by P(P_1=(a,b))
    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(i+1) - scipy.special.gammaln(n-i)
    term2 = scipy.special.gammaln(i+1)-scipy.special.gammaln(j+1)-scipy.special.gammaln(i-j+1)

    return term0 + term1 + term2 + i*np.log(p)+(n-1-i)*np.log(1-p)+i*np.log(0.5)

#  Probabilities for the case a=0, b>0
def get_logP1(b,n, eps0):

    q=np.exp(eps0)/(1+np.exp(eps0))
    p=np.exp(-eps0)

    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(b) - scipy.special.gammaln(n-b+1)

    return np.log(1-q) + (b-1)*np.log(p/2) + (n-b)*np.log(1-p) + term1



def get_omega(n, k, eps0, nx, L ,gamma):

    P1 = []
    Lx = []

    p_in = 1/(3+np.exp(eps0))

    q=np.exp(eps0)*p_in

    # parameter for C
    p=2*p_in

    p2=p_in/(1-np.exp(eps0)*p_in)

    # Throw away small tails, using Hoeffding
    tol_ = 42
    lower_i=int(max(0,np.floor((n-1)*(p-np.sqrt(tol_/(2*(n-1)))))))
    upper_i=int(min(n,np.ceil((n-1)*(p+np.sqrt(tol_/(2*(n-1)))))))

    print('total i: ' + str(upper_i-lower_i))

    for i in range(lower_i,upper_i):

        if i%100 == 0:
            print(i)

        lower_j=int(max(0,np.floor(i*(0.5-np.sqrt(tol_/(2*(i+1)))))))
        upper_j=int(min(i,np.ceil(i*(0.5+np.sqrt(tol_/(2*(i+1)))))))

        #for the cases b>0, a>0
        for j in range(lower_j,upper_j):
            p_temp=get_logP2(i,j,n,eps0,gamma)
            P1.append(p_temp)
            a=j+1
            b=i-j
            nom= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
            denom= q*(b/a)+p2*(1-q)*(1-q)  + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
            Lx.append(np.log(gamma*nom/denom + (1-gamma)))


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



    # Compute the periodisation & truncation error term,
    # using the error analysis given in
    # Koskela, Antti, et al.
    # Tight differential privacy for discrete-valued mechanisms and
    # for the subsampled gaussian mechanism using fft.
    # International Conference on Artificial Intelligence and Statistics. PMLR, 2021.

    # the parameter lambda in the error bound can be chosen freely.
    # commonly the error term becomes negligible for reasonable values of L.
    # here lambda is just hand tuned to make the error term small.

    lambd=0.01*L
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


    # check the mass to be sure
    print('Sum of Probabilities : ' + str(sum(omega_y)))

    return omega_y,grid_x, error_term



#Number of rounds
ks = [1,4,16]
#ks = [16]

gamma0=0.25

k0=4
eps0=np.log(((1-gamma0)*k0+gamma0)/gamma0)

gamma=1
nx=int(2E7)
L=100.0
n=int(1E3)



deltas=np.linspace(-4,-7,5)
deltas=10**deltas

epsilons=[]

for k in ks:

    omega,grid_x, error_term = get_omega(int(gamma*n), k, eps0,nx,L,gamma)

    print('k: ' + str(k))

    epsilons_temp=[]

    for delta_target in deltas:


        k_users = int(gamma*n)
        dx=2*L/nx

        half = int(nx/2)

        fx=np.copy(omega)

        # Compute the log MGFs and compute the error induced by the Fourier accountant,
        # using the analaysis provided by Koskela et al. (2021)

        # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
        temp = np.copy(fx[half:])
        fx[half:] = np.copy(fx[:half])
        fx[:half] = temp
        # Compute the DFT
        FF1 = np.fft.fft(fx)
        y = np.fft.ifft((FF1**k))


        # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
        temp = np.copy(y[half:])
        y[half:] = y[:half]
        y[:half] = temp
        omega1=y


        epsilon=0.1
        delta_eps=0.05
        y = omega1
        j_start = int(np.ceil((L+epsilon)/dx))
        delta_0 = np.real(np.sum((1-np.exp(epsilon-grid_x[j_start:]))*y[j_start:])) + error_term
        while abs(delta_0 - delta_target)>1e-9:

            if delta_0 < delta_target:
                epsilon = epsilon- delta_eps
                delta_eps = delta_eps/2
            else:
                epsilon = epsilon + delta_eps

            j_start = int(np.ceil((L+epsilon)/dx))
            delta_0 = np.real(np.sum((1 - np.exp(epsilon - grid_x[j_start:]))*y[j_start:])) + error_term
            print(str(delta_0)+ ' ' + str(epsilon))

        print('delta: ' + str(delta_target) + ' eps ' + str(epsilon))

        epsilons_temp.append(epsilon)

        pickle.dump(epsilons_temp, open('./pickles/eps_blanket_feldman_fa' + str(k) + '.p', "wb"))

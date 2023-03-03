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
def get_logP2(i,j,n, eps0):

    a=j+1
    b=i-j

    #mixture probability q = p exp(\veps_0)
    p_in = 1/(1+np.exp(eps0))

    q=np.exp(eps0)*p_in

    # parameter for C
    p=2*p_in

    p2=p_in/(1-np.exp(eps0)*p_in)


    term0 = np.log(q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a))
    # Then term0 is multiplied by P(P_0=(a,b))
    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(i+1) - scipy.special.gammaln(n-i)
    term2 = scipy.special.gammaln(i+1)-scipy.special.gammaln(j+1)-scipy.special.gammaln(i-j+1)

    return term0 + term1 + term2 + i*np.log(p)+(n-1-i)*np.log(1-p)+i*np.log(0.5)




def get_omega(n, k, eps0, nx, L):

    P1 = []
    Lx = []


    p_in = 1/(1+np.exp(eps0))

    q=np.exp(eps0)*p_in

    # parameter for C
    p=2*p_in

    p2=p_in/(1-np.exp(eps0)*p_in)

    tol_ = 60
    lower_i=int(max(0,np.floor((n-1)*(p-np.sqrt(tol_/(2*(n-1)))))))
    upper_i=int(min(n,np.ceil((n-1)*(p+np.sqrt(tol_/(2*(n-1)))))))

    for i in range(lower_i,upper_i):

        lower_j=int(max(0,np.floor(i*(0.5-np.sqrt(tol_/(2*(i+1)))))))
        upper_j=int(min(i,np.ceil(i*(0.5+np.sqrt(tol_/(2*(i+1)))))))

        #for the cases b>0, a>0
        for j in range(lower_j,upper_j):
            p_temp=get_logP2(i,j,n,eps0)
            P1.append(p_temp)
            a=j+1
            b=i-j
            nom= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
            denom= q*(b/a)+p2*(1-q)  + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
            Lx.append(np.log(nom/denom))


    dx=2*L/nx
    grid_x=np.linspace(-L,L-dx,nx)
    omega_y=np.zeros(nx)
    len_lx=len(Lx)
    for i in range(0,len_lx):
        lx=Lx[i]
        if lx>-L and lx<L:
            # place the mass to the right end point of the interval -> upper bound for delta
            ii=int(np.ceil((lx+L)/dx))
            omega_y[ii]=omega_y[ii]+np.exp(P1[i]);
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

    lambd=0.5*L
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


# parameters 1
n=int(1E4)
eps0=4.0
n_eps=60
epsilons = np.linspace(0.1,1.0,n_eps)




nx=int(1E7)

L=20

dx=2*L/nx

t1 = time.perf_counter()

k=1
omega,grid_x, error_term = get_omega(n, k, eps0,nx,L)
print('Time of forming PLD: ' + str(time.perf_counter()-t1))

delta_table=[]
deltas=[]

for eps in epsilons:
    y = omega
    j_start = int(np.ceil((L+eps)/dx))

    delta_temp = np.sum((1-np.exp(eps-grid_x[j_start:]))*y[j_start:])
    deltas.append(delta_temp)
    print('eps: ' + str(eps) + ' delta : ' + str(delta_temp))

delta_table.append(deltas)

ks=[2,3,4]


for k in ks:

    omega,grid_x, error_term = get_omega(n, k, eps0,nx,L)

    half = int(nx/2)

    fx=np.copy(omega)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    t1 = time.perf_counter()
    # Compute the DFT
    FF1 = np.fft.fft(fx)
    y = np.fft.ifft((FF1**k))
    print('Time of FFTs: ' + str(time.perf_counter()-t1))


    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(y[half:])
    y[half:] = y[:half]
    y[:half] = temp
    omega1=y

    deltas=[]

    for eps in epsilons:
        y = omega1
        j_start = int(np.ceil((L+eps)/dx))
        delta_temp = np.sum((1-np.exp(eps-grid_x[j_start:]))*y[j_start:])
        deltas.append(delta_temp)

    delta_table.append(deltas)


pickle.dump(delta_table, open("./pickles/deltas_pld.p", "wb"))

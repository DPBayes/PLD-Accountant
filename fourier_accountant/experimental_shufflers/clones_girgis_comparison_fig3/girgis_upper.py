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




def Y_term(gamma,eps0,k,lambda0):
    fact1 = (1+gamma*(np.exp(2*eps0)-1)/(2*np.exp(eps0)))**lambda0 - 1 - lambda0*gamma*(np.exp(2*eps0)-1)/(2*np.exp(eps0))
    fact2 = np.exp(-(k-1)/(8*np.exp(eps0)))
    return fact1*fact2

def log_lambda_over_j(lambda0,j):
    return scipy.special.gammaln(lambda0+1)-scipy.special.gammaln(j+1) - scipy.special.gammaln(lambda0+1-j)

def sum_term(gamma,eps0,k,j,lambda0):
    k_bar = np.floor((k-1)/(2*np.exp(eps0)))+1

    log_answer = log_lambda_over_j(lambda0,j) + j*np.log(gamma) + np.log(j) + scipy.special.gammaln(j/2) + (j/2)*np.log((2*(np.exp(2*eps0)-1)**2)/(k_bar*np.exp(2*eps0)) )

    return np.exp(log_answer)

def sum_eval(gamma,eps0,k,lambda0):
    summ=0
    for j in range(3,lambda0+1):
        summ+=sum_term(gamma,eps0,k,j,lambda0)
    return summ

def eval_RDP_upper(gamma,eps0,k,lambda0):
    k_bar = np.floor((k-1)/(2*np.exp(eps0)))+1
    summ=1+4*(lambda0*(lambda0-1)/2)*gamma**2*(np.exp(eps0)-1)**2/(k_bar*np.exp(eps0)) + sum_eval(gamma,eps0,k,lambda0) + Y_term(gamma,eps0,k,lambda0)
    return(np.log(summ)/(lambda0-1))


def eps_given_delta(RDPs,lambdas,delta):
    epsilons=[]
    for (ii,lambda0) in enumerate(lambdas):
        eps_value = RDPs[ii] + (np.log(1/delta) + (lambda0-1)*np.log(1-1/lambda0) - np.log(lambda0))/(lambda0-1)
        epsilons.append(eps_value)
    return(min(epsilons))

# numbers of compositions

Ts = [10,int(10**1.5),10**2,int(10**2.5),10**3,int(10**3.5),10**4]


n=int(1E5)

gamma=0.01

eps0=3.0

delta=1e-5

k=int(gamma*n)

epsilons=[]

for T in Ts:

    lambdas=np.linspace(2,1000,999)
    RDPs = []
    for lambda0 in lambdas:
        lambda0_int = int(lambda0)
        RDPs.append(T*eval_RDP_upper(gamma,eps0,k,lambda0_int))
    epsilon = eps_given_delta(RDPs,lambdas,delta)
    print(epsilon)

    epsilons.append(epsilon)

pickle.dump(epsilons, open("./pickles/eps_fig1_girgis_upper.p", "wb"))

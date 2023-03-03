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
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


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

    # The following gives the subsampled distribution
    #    gamma P + (1-gamma) Q
    #    = gamma ( q P_1 + (1-q) P_0 ) + (1-gamma) ( (1-q) P_1 + q P_0)

    # term0 = log(  fact1 + fact2*P(P_0=(a,b))/P(P_1=(a,b)))
    term0 = np.log(q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a))
    # Then term0 is multiplied by P(P_1=(a,b))
    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(i+1) - scipy.special.gammaln(n-i)
    term2 = scipy.special.gammaln(i+1)-scipy.special.gammaln(j+1)-scipy.special.gammaln(i-j+1)

    return term0 + term1 + term2 + i*np.log(p)+(n-1-i)*np.log(1-p)+i*np.log(0.5)



# Numerical RDPs, we use the Hoeffding bound to speed up the computation

def get_RDP(n, eps0, nx, L,alpha_max):

    P1 = []
    Lx = []


    p_in = 1/(1+np.exp(eps0))

    q=np.exp(eps0)*p_in

    # parameter for C
    p=2*p_in

    p2=p_in/(1-np.exp(eps0)*p_in)

    tol_ = 50
    lower_i=int(max(0,np.floor((n-1)*(p-np.sqrt(tol_/(2*(n-1)))))))
    upper_i=int(min(n,np.ceil((n-1)*(p+np.sqrt(tol_/(2*(n-1)))))))

    # lower_i=0
    # upper_i=n

    print('total i: ' + str(upper_i-lower_i))

    for i in range(lower_i,upper_i):

        if i%100 == 0:
            print(i)

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


    len_lx=len(Lx)
    print(len_lx)

    RDP=0

    max_order=alpha_max
    alphas = np.arange(2,max_order+1)
    RDPs = np.zeros(max_order-1)

    for i in range(0,len_lx):

        if i%1000==0:
            print(i)

        lx=Lx[i]



        for klm in range(max_order-1):
            RDPs[klm]+= np.exp(P1[i] + alphas[klm]*lx)

    # TODO: implement logsumexp here. Seems to work for this example without it.
    for klm in range(max_order-1):
        RDPs[klm] = np.log(RDPs[klm])/(alphas[klm]-1)

    return RDPs,alphas


# parameters 1
n=int(1E4)
eps0=4.0
n_eps=60
epsilons = np.linspace(0.1,1.0,n_eps)




nx=int(1E7)

L=20

dx=2*L/nx

t1 = time.perf_counter()


RDPs,alphas = get_RDP(n, eps0,nx,L,alpha_max=256)

print(RDPs)

print('Time of computing RDP: ' + str(time.perf_counter()-t1))

RDPs = np.array(RDPs)


# Use the Cannone et al. (2020) - formula for the RDP -> (eps,delta)-conversion

deltas=[]

for eps_ref in epsilons:
    delta=min((np.exp((alphas-1)*(RDPs-eps_ref))/alphas)*((1-1/alphas)**(alphas-1)))
    deltas.append(delta)

print(deltas)


deltas2=[]

for eps_ref in epsilons:
    delta=min((np.exp((alphas-1)*(2*RDPs-eps_ref))/alphas)*((1-1/alphas)**(alphas-1)))
    deltas2.append(delta)

deltas3=[]

for eps_ref in epsilons:
    delta=min((np.exp((alphas-1)*(3*RDPs-eps_ref))/alphas)*((1-1/alphas)**(alphas-1)))
    deltas3.append(delta)

deltas4=[]

for eps_ref in epsilons:
    delta=min((np.exp((alphas-1)*(4*RDPs-eps_ref))/alphas)*((1-1/alphas)**(alphas-1)))
    deltas4.append(delta)


pickle.dump(deltas, open("./pickles/deltas_feldman.p", "wb"))
pickle.dump(deltas2, open("./pickles/deltas_feldman2.p", "wb"))
pickle.dump(deltas3, open("./pickles/deltas_feldman3.p", "wb"))
pickle.dump(deltas4, open("./pickles/deltas_feldman4.p", "wb"))

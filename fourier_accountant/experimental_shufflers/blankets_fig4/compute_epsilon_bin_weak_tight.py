"""
Computing tight DP guarantees for the weak adversary
"""

import numpy as np
import scipy
import scipy.special
import math







# get the privacy loss distribution
def get_omega(n, gamma,k, nx=int(1E6), L=1000):


    # Probabilities and values for B (B ~ Bin(n-1,gamma))

    Blog=[]

    log_inv_tol=40
    i_min = int(max(0,np.floor((n-1)*(gamma-np.sqrt((np.log(gamma) + log_inv_tol)/(2*(n-1)))))))
    i_max = int(min(n,np.ceil((n)*(gamma+np.sqrt((np.log(gamma) + log_inv_tol)/(2*(n-1)))))))

    # i_min=0
    # i_max=n

    for i in range(i_min,i_max+1):
        Blog.append(scipy.special.gammaln(n)-scipy.special.gammaln(n-i)-scipy.special.gammaln(i+1) + i*np.log(gamma) + (n-1-i)*np.log(1-gamma))

    print('i max - i min : ' + str(i_max - i_min))

    B = np.exp(np.array(Blog))

    print(' B sum : ' + str(np.sum(B)))

    p=1/k
    p2=1/(k-1)

    Wz = []
    Fz = []

    sum_nBN1N2=0

    sum_Wz_zero=0

    for nB in range(i_min,i_max+1):

        # Probabilities for N_1|B (N_1|B ~ Bin(B,1/k) + Bern...)
        tot_prob_per_B=0
        N1=np.zeros(nB+2)
        N2s=[]
        for i in range(0,nB+1):

            Ntemp = scipy.special.gammaln(nB+1)-scipy.special.gammaln(nB-i+1)-scipy.special.gammaln(i+1) + i*np.log(p) + (nB-i)*np.log(1-p)
            N1[i] += (gamma-gamma/k)*np.exp(Ntemp)

            Ntemp = scipy.special.gammaln(nB+1)-scipy.special.gammaln(nB-i+1)-scipy.special.gammaln(i+1) + i*np.log(p) + (nB-i)*np.log(1-p)
            N1[i+1] += (1-gamma+gamma/k)*np.exp(Ntemp)

            # Probabilities for N_2|B (N_2|B ~ Bin(B-i,1/(k-1)) + Bern(gamma/k))
            N2=np.zeros(nB-i+2)
            for j in range(0,nB-i+1):
                Ntemp = scipy.special.gammaln((nB+1)-i+1)-scipy.special.gammaln((nB+1)-i-j+1)-scipy.special.gammaln(j+1) + j*np.log(p2) + ((nB+1)-i-j)*np.log(1-p2)
                N2[j] += np.exp(Ntemp)
                tot_prob_per_B+= np.exp(Ntemp)

            N2s.append(N2)


        for i in range(0,nB+1):
            for j in range(0,nB-i+1):
                nom = (1-gamma)*i + (gamma/k)*nB
                denom = (1-gamma)*j + (gamma/k)*nB

                sum_nBN1N2 += B[nB-i_min]*N1[i]*N2s[i][j]
                if denom != 0:# and nom != 0:
                    Wz.append(np.log(nom/denom))
                    Fz.append(B[nB-i_min]*N1[i]*N2s[i][j])
                else:
                    sum_Wz_zero+=B[nB-i_min]*N1[i]*N2s[i][j]


    print(' sum_nBN1N2 : ' + str(sum_nBN1N2))
    print('len(Wz) : ' + str(len(Wz)))

    print(' sum Fz: ' + str(np.sum(np.array(Fz))))

    dx=2*L/nx
    grid_x=np.linspace(-L,L-dx,nx)
    omega_y=np.zeros(nx)
    for i in range(1,len(Wz)):
        lx=Wz[i]
        if lx>-L and lx<L:
            # place the mass to the right end point of the interval -> upper bound for delta
            ii=int(np.ceil((lx+L)/dx))
            omega_y[ii]=omega_y[ii]+Fz[i];




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

    for i in range(len(Wz)):
        plf = -Wz[i]
        lambda_sum_minus+=np.exp(lambd*plf + Fz[i])

    alpha_minus=np.log(lambda_sum_minus)

    print(alpha_minus)

    #Then alpha plus
    lambda_sum_plus=0
    for i in range(len(Wz)):
        plf = -Wz[i]
        lambda_sum_plus+=np.exp(lambd*plf + Fz[i])

    alpha_plus=np.log(lambda_sum_plus)

    #Evaluate the periodisation error bound using alpha_plus and alpha_minus
    T1=(np.exp(k*alpha_plus)  )
    T2=(np.exp(k*alpha_minus) )
    error_term = (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-2*lambd*L)))
    print('error_term: ' + str(error_term))

    print('sum omega: ' + str(np.sum(omega_y)))

    return omega_y,grid_x, error_term








def get_epsilons(omega_y,error_term,grid_x,L,nx, target_delta=1e-6):




    dx=2*L/nx

    # Len_Lx=len(Lx)
    # dd=1e-6

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

        #print(eps)


        ssi=int(np.ceil((epsm+L)/dx))
        sum_delta_m=np.real(np.sum((1-np.exp(epsm-grid_x[ssi:]))*omega_y[ssi:])) + error_term

        ssi=int(np.ceil((eps0+L)/dx))
        sum_delta_0=np.real(np.sum((1-np.exp(eps0-grid_x[ssi:]))*omega_y[ssi:])) + error_term



        if math.copysign(1, sum_delta_m - dd) == math.copysign(1, sum_delta_0 - dd):
            eps0 = epsm
        else:
            eps1 = epsm

        epsm=(eps1+eps0)/2
        d_eps=(eps1-eps0)/2
        print(epsm)

    return eps0

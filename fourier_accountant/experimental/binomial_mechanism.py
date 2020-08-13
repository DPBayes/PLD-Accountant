"""
Experimental implementation for computing tight DP guarantees for the binomial mechanism.
The method is described in the manuscript
Tight Approximate Differential Privacy for Discrete-Valued Mechanisms Using FFT .
"""

import numpy as np
import scipy
import scipy.special

def get_epsilon(n, p, s, f_diff, target_delta=1e-4):
    """
    Computes epsilon given a target value for delta for the binomial mechanism.

    Args:
        n: Parameter n of binomial mechanism
        p: Parameter p of binomial mechanism
        s: Parameter s of binomial mechanism
        f_diff: Difference of output between datasets of underlying function f. \Delta = f(X) - f(Y) in Thm. 11. Here assumed constant for all elements.
        target_delta: Target delta to compute epsilon for.
    """
    D2 = int(1./(s*f_diff))

    B2=np.zeros(n+1)
    for i in range(0,n+1):
        B2[i] = scipy.special.gammaln(n+1)-scipy.special.gammaln(n-i+1)-scipy.special.gammaln(i+1) + i*np.log(p) + (n-i)*np.log(1-p)
    B=B2.copy()

    # Compute the logarithmic ratios,
    # i.e. s_i's for \omega_{X/Y}
    Lx=np.zeros(n+1-D2)
    for i in range(D2,n+1):
        Lx[i-D2]=B[i]-B[i-D2]

    # Compute the logarithmic ratios,
    # i.e. s_i's for \omega_{Y/X}
    Ly=np.zeros(n+1-D2)
    for i in range(D2,n+1):
        Ly[i-D2] = -(B[i]-B[i-D2])


    Wx=np.exp(B[D2:])
    Wy=np.exp(B[:-D2])

    # delta_inf terms
    delta_inf_XY = np.sum(np.exp(B[:D2]))
    delta_inf_YX = np.sum(np.exp(B[-D2:]))

    print(' delta_inf_XY :' + str(delta_inf_XY))

    neps=10
    epsilons=np.linspace(1.0,2.0,neps)
    deltas=np.zeros(neps)
    eps=1.0
    Len_Lx=len(Lx)
    Len_Ly=len(Ly)
    dd=target_delta
    sum_deltax=1.0
    sum_deltay=1.0
    tol=1e-10
    delta_eps=1.0

    # Run binary search to find eps(delta)
    while abs(dd-sum_deltax)>tol:

        if sum_deltax>dd:
            eps=eps+delta_eps
            if delta_eps<1:
                delta_eps/=2
        else:
            eps=eps-delta_eps
            delta_eps/=2

        #Add the delta_inf term
        sum_deltax=delta_inf_XY
        for j in range(0,Len_Lx):
            if Lx[j]>=eps:
                sum_deltax=sum_deltax + (1-np.exp(eps-Lx[j]))*Wx[j]

    epsx=eps

    # Run binary search to find eps(delta)
    eps=1.0
    delta_eps=1.0
    while abs(dd-sum_deltay)>tol:
        if sum_deltay>dd:
            eps=eps+delta_eps
            if delta_eps<1:
                delta_eps/=2
        else:
            eps=eps-delta_eps
            delta_eps/=2

        #Add the delta_inf term
        sum_deltay=delta_inf_YX
        for j in range(0,Len_Ly):
            if Ly[j]>=eps:
                sum_deltay=sum_deltay + (1-np.exp(eps-Ly[j]))*Wy[j]

    epsy=eps

    eps_bin = max(epsx,epsy)
    s_bin = s*np.sqrt(0.01*n*p*(1-p))

    return eps_bin, s_bin





import numpy as np
import scipy
import scipy.special
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib
import pickle



#  Probabilities for the case a>0
def get_logP2(i,j,n, eps0,gamma):



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


    P= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
    Q= q*(b/a)+p2*(1-q)  + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
    # term0 = log(  fact1 + fact2*P(P_0=(a,b))/P(P_1=(a,b)))
    term0 = np.log(gamma*P+(1-gamma)*Q)
    # Then term0 is multiplied by P(P_1=(a,b))
    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(i+1) - scipy.special.gammaln(n-i)
    term2 = scipy.special.gammaln(i+1)-scipy.special.gammaln(j+1)-scipy.special.gammaln(i-j+1)

    return term0 + term1 + term2 + i*np.log(p)+(n-1-i)*np.log(1-p)+i*np.log(0.5)



def get_logP4(i,j,n, eps0,gamma):

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


    P= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/(2*(1-2*p)*a)
    # term0 = log(  fact1 + fact2*P(P_0=(a,b))/P(P_1=(a,b)))
    term0 = np.log(P)
    # Then term0 is multiplied by P(P_1=(a,b))
    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(i+1) - scipy.special.gammaln(n-i)
    term2 = scipy.special.gammaln(i+1)-scipy.special.gammaln(j+1)-scipy.special.gammaln(i-j+1)

    return term0 + term1 + term2 + i*np.log(p)+(n-1-i)*np.log(1-p)+i*np.log(0.5)




def get_omega(n, eps0, nx, L ,gamma):

    P1 = []
    Lx = []

    p_in = 1/(1+np.exp(eps0))

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

    # check the mass to be sure
    print('Sum of Probabilities : ' + str(sum(omega_y)))

    return omega_y,grid_x



def get_omega2(n, eps0, nx, L ,gamma):

    P1 = []
    Lx = []

    p_in = 1/(1+np.exp(eps0))

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
            p_temp=get_logP4(i,j,n,eps0,gamma)
            P1.append(p_temp)
            a=j+1
            b=i-j
            nom= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
            denom= q*(b/a)+p2*(1-q)  + (1-p2)*(1-q)*(n-(a+b))*p/((1-2*p)*a)
            Lx.append(-np.log(gamma*denom/nom + (1-gamma)))


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

    return omega_y,grid_x











def get_PQ2(omega,omega2,nx=2E6,L=30.0):


    nx = len(omega)

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    fx = omega/np.sum(omega)
    fx2 = omega2/np.sum(omega2)

    cfx=fx
    cfx2=fx2

    n_alphas=3000

    alphas_log =np.linspace(-.2,.2,n_alphas)
    alphas=np.exp(alphas_log)

    Q = np.zeros(n_alphas)
    P = np.zeros(n_alphas)


    for ijkl in range(1,n_alphas-1):

        print(ijkl)

        hminus=0
        h=0
        hplus=0

        # h_minus

        if ijkl==1:
            hminus=1
            alphaminus=0
        else:
            alphaminus=alphas[ijkl-1]
            target_eps=np.log(alphaminus)

            jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))
            # Evaluate \delta(target_eps) and \delta'(target_eps)
            exp_e = 1-np.exp(target_eps-x[jj+1:])

            integrand1 = exp_e*cfx[jj+1:]
            integrand2 = exp_e*cfx2[jj+1:]

            sum_int1=np.sum(integrand1)
            sum_int2=np.sum(integrand2)
            hminus = max(sum_int1,sum_int2)

        print('hminus' + str(hminus) + ' alpha ' + str(alphaminus))

        alpha=alphas[ijkl]
        target_eps=np.log(alpha)

        jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))
        # Evaluate \delta(target_eps) and \delta'(target_eps)
        exp_e = 1-np.exp(target_eps-x[jj+1:])


        integrand1 = exp_e*cfx[jj+1:]
        integrand2 = exp_e*cfx2[jj+1:]

        sum_int1=np.sum(integrand1)
        sum_int2=np.sum(integrand2)
        h = max(sum_int1,sum_int2)

        print('h' + str(h) + ' alpha ' + str(alpha))

        alphaplus=alphas[ijkl+1]
        target_eps=np.log(alphaplus)

        jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))
        # Evaluate \delta(target_eps) and \delta'(target_eps)
        exp_e = 1-np.exp(target_eps-x[jj+1:])

        integrand1 = exp_e*cfx[jj+1:]
        integrand2 = exp_e*cfx2[jj+1:]

        sum_int1=np.sum(integrand1)
        sum_int2=np.sum(integrand2)
        hplus = max(sum_int1,sum_int2)

        print('hplus' + str(hplus) + ' alpha ' + str(alphaplus))

        Q[ijkl] = (hminus-h)/(alpha-alphaminus) - (h-hplus)/(alphaplus-alpha)
        P[ijkl] = alpha*Q[ijkl]

        if ijkl==n_alphas-2:
            Q[ijkl] = (hminus-h)/(alpha-alphaminus)
            P[ijkl] = alpha*Q[ijkl]

    Q[0] = 1-np.sum(Q)
    P[0] = 0

    P[-1] = 0


    Q=np.abs(Q)
    P=np.abs(P)

    print(Q)
    print(P)
    print('sum of Qs : ' + str(np.sum(Q)))
    print('sum of Ps : ' + str(np.sum(P)))

    return P,Q


def get_delta_orig(omega,grid_x,max_eps,ncomp=1E4,nx=3E6,L=30.0):

    nx = int(len(grid_x))

    x = grid_x # grid for the numerical integration

    fx = omega/np.sum(omega)

    half = int(nx/2)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    # Compute the DFT
    FF1 = np.fft.fft(fx)

    # Compute the inverse DFT
    cfx = np.fft.ifft((FF1**ncomp))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    n_eps=40
    epsilons = np.linspace(0.2,max_eps,n_eps)

    deltas=[]

    for target_eps in epsilons:

        jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))
        # Evaluate \delta(target_eps) and \delta'(target_eps)
        exp_e = 1-np.exp(target_eps-x[jj+1:])
        integrand = exp_e*cfx[jj+1:]
        sum_int=np.sum(integrand)
        delta = sum_int

        deltas.append(np.real(delta))
        print(np.real(delta))

    return deltas





def get_delta_PQ(P,Q,max_eps,ncomp=1E3,nx=3E6,L=30.0):

    nx = int(nx)

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    Wx = P
    Wy = Q

    Lx = np.log(Wx/Wy)

    dx=2*L/nx
    grid_x=x
    omega_y=np.zeros(nx)
    len_lx=len(Lx)
    for i in range(0,len_lx):
        lx=Lx[i]
        if lx>-L and lx<L:
            ii=int(round((lx+L)/dx))
            omega_y[ii]=omega_y[ii]+Wx[i];
    print(sum(omega_y))

    fx = omega_y/np.sum(omega_y)

    dx=1.0

    half = int(nx/2)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    # Compute the DFT
    FF1 = np.fft.fft(fx*dx)

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain


    # Compute the inverse DFT
    cfx = np.fft.ifft((FF1**ncomp/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    n_eps=40
    epsilons = np.linspace(0.2,max_eps,n_eps)

    deltas=[]

    for target_eps in epsilons:

        jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))
        # Evaluate \delta(target_eps) and \delta'(target_eps)
        exp_e = 1-np.exp(target_eps-x[jj+1:])
        integrand = exp_e*cfx[jj+1:]
        sum_int=np.sum(integrand)
        delta = sum_int*dx

        deltas.append(np.real(delta))
        print(np.real(delta))

    return deltas





def get_epsilon(omega,omega2,delta_target,ncomp=1E3,nx=3E6,L=30.0):




    fx = omega/np.sum(omega)
    fx2 = omega2/np.sum(omega2)

    dx=1.0
    half = int(nx/2)


    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    # Compute the DFT
    FF1 = np.fft.fft(fx*dx)

    # Compute the inverse DFT
    cfx = np.fft.ifft((FF1**ncomp/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp


    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx2[half:])
    fx2[half:] = np.copy(fx2[:half])
    fx2[:half] = temp
    # Compute the DFT
    FF1 = np.fft.fft(fx2*dx)
    # Compute the inverse DFT
    cfx2 = np.fft.ifft((FF1**ncomp/dx))
    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx2[half:])
    cfx2[half:] = cfx2[:half]
    cfx2[:half] = temp

    eps0=1.0
    delta_eps=eps0/2
    y = cfx
    y2 = cfx2
    j_start = int(np.ceil((L+eps0)/dx))
    delta_1 = np.real(np.sum((1-np.exp(eps0-grid_x[j_start:]))*y[j_start:]))
    delta_2 = np.real(np.sum((1-np.exp(eps0-grid_x[j_start:]))*y2[j_start:]))
    delta_0 = max(delta_1,delta_2)
    while abs(delta_0 - delta_target)>1e-7:

        if delta_0 < delta_target:# and delta_0 > 0:
            eps0=eps0- delta_eps
            delta_eps=delta_eps/2
        else:
            eps0=eps0+delta_eps

        print('delta0: ' + str(delta_0) + ' delta_target: ' + str(delta_target) + 'eps: ' + str(eps0) + ' error: ' + str(abs(delta_0 - delta_target)))
        j_start = int(np.ceil((L+eps0)/dx))

        delta_1 = np.real(np.sum((1-np.exp(eps0-grid_x[j_start:]))*y[j_start:]))
        delta_2 = np.real(np.sum((1-np.exp(eps0-grid_x[j_start:]))*y2[j_start:]))
        delta_0 = max(delta_1,delta_2)


    return eps0




delta_target=1e-5

epsilons=[]

ks = [10,int(10**1.5),10**2,int(10**2.5),10**3,int(10**3.5),10**4]


eps0=3
gamma=0.01
nx_=int(1E6)
L_=10.0
n=int(1E5)

n_gamma = int(gamma*n)
omega,grid_x = get_omega(n_gamma, eps0,nx_,L_,gamma)
omega2,grid_x = get_omega2(n_gamma, eps0,nx_,L_,gamma)

n_eps=40

max_eps=6.0
#
nc=1500
#
epsilons = np.linspace(0.2,max_eps,n_eps)


epsilons=[]

for k in ks:

    eps0=get_epsilon(omega,omega2,delta_target,ncomp=k,nx=nx_,L=L_)
    epsilons.append(eps0)
    print('k: ' + str(k) + ' eps: ' + str(eps0))


pickle.dump(epsilons, open("./pickles/eps_fig1_addremove_pld.p", "wb"))

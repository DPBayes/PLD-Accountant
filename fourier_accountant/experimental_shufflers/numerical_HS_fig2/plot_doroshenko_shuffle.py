




import numpy as np
import scipy
import scipy.special
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib



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

    P= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/(2*(1-2*p)*a)
    Q= q*(b/a)+p2*(1-q)  + (1-p2)*(1-q)*(n-(a+b))*p/(2*(1-2*p)*a)
    # term0 = log(  fact1 + fact2*P(P_0=(a,b))/P(P_1=(a,b)))
    term0 = np.log(gamma*P+(1-gamma)*Q)
    # Then term0 is multiplied by P(P_1=(a,b))
    term1 = scipy.special.gammaln(n)-scipy.special.gammaln(i+1) - scipy.special.gammaln(n-i)
    term2 = scipy.special.gammaln(i+1)-scipy.special.gammaln(j+1)-scipy.special.gammaln(i-j+1)

    return term0 + term1 + term2 + i*np.log(p)+(n-1-i)*np.log(1-p)+i*np.log(0.5)




#  Probabilities for the case a>0
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



# return the PLD   log(P(t)/Q(t)), t ~ P
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
            nom= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/(2*(1-2*p)*a)
            denom= q*(b/a)+p2*(1-q)*(1-q)  + (1-p2)*(1-q)*(n-(a+b))*p/(2*(1-2*p)*a)
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


# return the PLD   log(Q(t)/P(t)), t ~ Q
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
            nom= q + p2*(1-q)*(b/a) + (1-p2)*(1-q)*(n-(a+b))*p/(2*(1-2*p)*a)
            denom= q*(b/a)+p2*(1-q)  + (1-p2)*(1-q)*(n-(a+b))*p/(2*(1-2*p)*a)
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





# return the dominating pair of distributions (P,Q)
# (for the subsampled mechanism, withour replacement sampling, substitute relation)

def get_PQ2(n_alphas,omega,omega2,nx=2E6,L=30.0):

    nx = len(omega)

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    fx = omega/np.sum(omega)
    fx2 = omega2/np.sum(omega2)

    cfx=fx
    cfx2=fx2


    alphas_log =np.linspace(-.5,.5,n_alphas)
    alphas=np.exp(alphas_log)


    #alphas=np.linspace(0.001,np.exp(3),n_alphas)

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




# the bound computed only with the PLD log(P(t)/Q(t)), t ~ P
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

    print('sum: ' + str(np.sum(Wx)))

    dx=2*L/nx
    grid_x=x
    omega_y=np.zeros(nx)
    len_lx=len(Lx)
    for i in range(0,len_lx):
        lx=Lx[i]
        if lx>-L and lx<L:
            ii=int(np.ceil((lx+L)/dx))
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

n_eps=40
max_eps=1.13
nc=2000
epsilons = np.linspace(0.2,max_eps,n_eps)

eps0=3
gamma=0.01
nx_=int(4E6)
L_=20.0
n=int(1E4)

omega,grid_x = get_omega(n, eps0,nx_,L_,gamma)
omega2,grid_x = get_omega2(n, eps0,nx_,L_,gamma)

n_a=[500,3000]

delta_table_PQ=[]
delta_table_orig=[]

for n_alphas in n_a:

    P,Q = get_PQ2(n_alphas,omega,omega2,nx=nx_,L=L_)
    deltas_PQ=get_delta_PQ(P,Q,max_eps,ncomp=nc,nx=nx_,L=L_)
    delta_table_PQ.append(deltas_PQ)


deltas_orig=get_delta_orig(omega,grid_x,max_eps,ncomp=nc,nx=nx_,L=L_)



pp = PdfPages('./plots/deltas_doro3.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 15.0})
plt.rc('text', usetex=True)
plt.rc('font', family='arial')

plt.ylabel('$\delta$')
plt.xlabel(r'$\varepsilon$')

Ncols = 8
colors = [matplotlib.cm.viridis(x) for x in np.linspace(0, 0.8, Ncols)]

for (ii,deltas_PQ) in enumerate(delta_table_PQ):
    plt.semilogy(epsilons,deltas_PQ,'--',color=colors[2*ii],linewidth=1)

plt.semilogy(epsilons,deltas_orig,'-',color=colors[5],linewidth=1)

legs=[]
legs.append(r'Doroshenko et al., $n_\alpha = 500$')
legs.append(r'Doroshenko et al., $n_\alpha = 3000$')
legs.append(r'$H_\alpha(q \cdot P + (1-q) \cdot Q||Q)$')
plt.legend(legs,loc='lower left')



pp.savefig(plot_, bbox_inches = 'tight', pad_inches = 0)
pp.close()


plt.show()
plt.close()

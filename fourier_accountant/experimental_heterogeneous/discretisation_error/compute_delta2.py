

import numpy as np
import scipy
import scipy.special
import math

def get_deltas(n1, p, D2, n_comp, nx=int(4E6), L=1000, target_eps=1.0):

    n=D2*n1
    Blog=np.zeros(n+1)

    # Compute the log probabilities, for numerical stability
    for i in range(0,n+1):
        Blog[i] = scipy.special.gammaln(n+1)-scipy.special.gammaln(n-i+1)-scipy.special.gammaln(i+1) + i*np.log(p) + (n-i)*np.log(1-p)

    B = np.exp(Blog)
    Wx=np.zeros(len(B)+D2);
    Wy=np.zeros(len(B)+D2);

    Wx[D2:]=B
    Wy[:len(B)]=B

    Lx=np.zeros(len(B))
    #for i in range(D2,len(Wx)):
    for i in range(D2,len(B)):
        Lx[i]=Blog[i-D2] - Blog[i]
        #np.log(Wx[i]) - np.log(Wy[i])
        # if math.isnan(Lx[i]):
        #     Lx[i]=-2*L
        # if Lx[i] is math.inf:
        #     Lx[i] = -2*L

    Ly=np.zeros(len(Wy))
    for i in range(D2,len(B)):
        Ly[i] = Blog[i] - Blog[i-D2]



    deltas=np.zeros(n_comp)

    dx=2*L/nx
    grid_x=np.linspace(-L,L-dx,nx)
    omega_y=np.zeros(nx)
    len_lx=len(Lx)
    for i in range(D2,len(B)):
        lx=Lx[i]
        # if math.isnan(lx):
        #     print('isnan: ' + str(i))

        if lx>-L and lx<L:
            #ii=int(round((lx+L)/dx))
            ii=int(np.ceil((lx+L)/dx))
            omega_y[ii]=omega_y[ii]+Wx[i];

    # for i in range(0,len(Wx)):
    #     if math.isnan(Lx[i]):
    #         print('Lx isnan: ' + str(i))
    #
    # for i in range(0,len(Wy)):
    #     if math.isnan(Ly[i]):
    #         print('Ly isnan: ' + str(i))

    eps=target_eps


    k=n_comp
    # Compute the periodisation & truncation error term
    lambd=2*L

    #First alpha_minus
    lambda_sum_minus=0
    #for i in range(D2,len(B)):
    #    lambda_sum_minus+=(Wy[i]/Wx[i])**lambd*Wy[i]
    for i in range(D2,len(B)):
        lambda_sum_minus+=np.exp(lambd*Ly[i])*Wy[i]

    #print('aminus' + str(lambda_sum_minus))
    alpha_minus=np.log(lambda_sum_minus)

    #Then alpha plus
    lambda_sum_plus=0
    for i in range(D2,len(B)):
        lambda_sum_plus+=np.exp(lambd*Lx[i])*Wx[i]
    #print('aplus' + str(lambda_sum_plus))
    alpha_plus=np.log(lambda_sum_plus)

    #Evaluate the periodisation error bound using alpha_plus and alpha_minus
    T1=(np.exp(k*alpha_plus)  )
    T2=(np.exp(k*alpha_minus) ) #/(np.exp(alpha_minus) - 1)
    ep = (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-2*lambd*L)))
    #print('periodisation error: ' + str(ep))



    lambdas=np.linspace(0.1,7.0,70)
    lambdas=L*lambdas
    #[0.25*L,0.5*L,1.0*L,2.0*L,3.0*L,4.0*L]
    eds=[]
    for ll in lambdas:
        #Evaluate the discretisation error bound
        lambd=ll
        #Then alpha plus
        lambda_sum_plus=0
        for i in range(D2,len(B)):
            lambda_sum_plus+=np.exp(lambd*Lx[i])*Wx[i]
        alpha_plus=np.log(lambda_sum_plus)
        ed=k*dx*np.exp(k*alpha_plus)*np.exp(-lambd*eps)
        #print('discretisation error: ' + str(ed))
        eds.append(ed)
    ed=min(eds)

    #Evaluate the convolutions of PLDs using FFT
    half = int(nx/2)

    fx=np.copy(omega_y)
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


    ssi=int(np.ceil((eps+L)/dx))
    dd=np.sum((1-np.exp(eps-grid_x[ssi:]))*y[ssi:])
    #print('sum_deltax : ' + str(dd))

    #print( ('%1.1f' % target_eps) + '  &  ' + ('%10.3E' % ep) + ' & ' + ('%10.3E' % ed) + ' & ' + ('%10.3E' % dd) + '\\\\')
    print( ('%1.1f' % target_eps) + ' & ' + ('%10.2E' % ed) + ' & ' + ('%10.5E' % dd) + '\\\\')
    return deltas

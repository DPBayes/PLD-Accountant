
import numpy as np
from scipy import optimize

def f(x, sigma, q):

    Linvx = (sigma**2)*np.log((np.exp(x)-(1-q))/q) + 0.5
    ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
    	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)))
    dLinvx = sigma**2*np.exp(x)/(np.exp(x)-(1-q))
    omega=ALinvx*dLinvx
    return omega

def df(x, sigma, q):

    Linvx = (sigma**2)*np.log((np.exp(x)-(1-q))/q) + 0.5
    ALinvx = ((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
    	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)))
    dALinvx = -2*Linvx*(1-q)*np.exp(-Linvx*Linvx/(2*sigma**2))/(2*sigma**2) - 2*(Linvx-1)*q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2))/(2*sigma**2)
    dLinvx = sigma**2*np.exp(x)/(np.exp(x)-(1-q))
    ddLinvx = -(1-q)*sigma**2*np.exp(x)/((np.exp(x)-(1-q))**2)
    domega=dALinvx*(dLinvx**2) + ALinvx*ddLinvx
    return abs(domega)

def get_pld(sigma=2.0,q=0.01,nx=1E6,L=20.0):
    #Find the peak point of the PLD
    xd0 = optimize.fmin(df,x0=0,args=(sigma,q),xtol=1e-9,disp=False)
    #optimize.options(disp=False)
    fdd0 = f(xd0,sigma,q)
    nx = int(nx)
    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration
    # first ii for which x(ii)>log(1-q),
    # i.e. start of the integral domain
    ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))
    # Evaluate the PLD distribution,
    # The case of remove/add relation (Subsection 5.1)
    Linvx = (sigma**2)*np.log((np.exp(x[ii+1:])-(1-q))/q) + 0.5
    ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
    	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
    dLinvx = (sigma**2*np.exp(x[ii+1:]))/(np.exp(x[ii+1:])-(1-q))
    fx_ref = np.zeros(nx)
    fx_ref[ii+1:] =  np.real(ALinvx*dLinvx)
    fx = np.zeros(nx)
    #Majorant for fx, be careful around the peak
    jj = int(np.ceil(float(nx*(L+xd0)/(2*L))))
    fx[:jj]=fx_ref[:jj]
    fx[jj] = fdd0
    fx[jj+1:]=fx_ref[jj+1:]
    return dx*fx

def get_delta_max(target_eps=1.0,sigmas=[2.0,2.0],q=0.01,ncomp=100,nx=1E6,L=20.0):


    ncomp_total = len(sigmas)*ncomp

    nx = int(nx)
    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    plds=[]
    for ss in sigmas:
        plds.append(get_pld(sigma=ss,q=q,nx=nx,L=L))

    error_t=0.0
    for fx in plds:
        #Compute the log MGFs
        lambd=L/2
        alpha_plus=np.log(np.sum(np.exp(lambd*x)*fx))
        alpha_minus=np.log(np.sum(np.exp(lambd*np.flip(x))*np.flip(fx)))

        T1=(2*np.exp((ncomp_total+1)*alpha_plus) - np.exp((ncomp_total)*alpha_plus) - np.exp(alpha_plus) )/(np.exp(alpha_plus - 1))
        T2=(np.exp((ncomp_total+1)*alpha_minus) - np.exp(alpha_minus) )/(np.exp(alpha_minus - 1))
        error_term= (T1+T2)*(np.exp(-lambd*L)/(1-np.exp(-lambd*L)))
        if error_term > error_t:
            error_t = error_term

    half = int(nx/2)

    FFTs = []
    for fx in plds:
        # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
        temp = np.copy(fx[half:])
        fx[half:] = np.copy(fx[:half])
        fx[:half] = temp

        # Compute the DFT
        FF1 = np.fft.fft(fx)
        FFTs.append(FF1)

    cfx = np.ones(nx)
    for FF in FFTs:
        # Compute the inverse DFT
        cfx = cfx*(FF**ncomp)

    cfx = np.fft.ifft(cfx)
    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+target_eps)/(2*L))))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1-np.exp(target_eps-x[jj+1:])
    integrand = exp_e*cfx[jj+1:]
    sum_int=np.sum(integrand)
    delta = sum_int + error_term

    #print('DP-delta (in R-relation) after ' + str(int(ncomp_total)) + ' compositions:' + str(np.real(delta)) + ' (epsilon=' + str(target_eps) + ')')

    return np.real(delta)







'''
A code for computing exact DP guarantees.
The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Exact Guarantees for Differential Privacy.
arXiv preprint arXiv:1906.03049 (2019)
The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko)
'''




import numpy as np



# Parameters:
# target_delta - target delta
# sigma_t - array of sigma values
# q_t - array of q values
# nx - number of points in the discretisation grid
# L -  limit for the integral

def get_eps_unbounded(sigma_t,q_t,target_delta=1e-6,nx=1E6,L=20.0):

    nx = int(nx)

    tol_newton = 1e-10 # set this to, e.g., 0.01*target_delta

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration


    fx_table=[]
    F_prod=np.ones(x.size)

    ncomp=sigma_t.size

    if(q_t.size != ncomp):
        print('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):

        sigma=sigma_t[ij]
        q=q_t[ij]

        # first ii for which x(ii)>log(1-q),
        # i.e. start of the integral domain
        ii = int(np.floor(float(nx*(L+np.log(1-q))/(2*L))))

        # Evaluate the PLD distribution,
        # The case of remove/add relation (Subsection 5.1)
        Linvx = (sigma**2)*np.log((np.exp(x[ii-1:])-(1-q))/q) + 0.5
        ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2)) +
        	q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)));
        ey = np.exp(x[ii-1:])
        dLinvx = (sigma**2)/(1-(1-q)/ey);

        fx = np.zeros(nx)
        fx[ii-1:] =  np.real(ALinvx*dLinvx)
        half = int(nx/2)

        # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
        temp = np.copy(fx[half:])
        fx[half:] = np.copy(fx[:half])
        fx[:half] = temp

        # Compute the DFT
        FF1 = np.fft.fft(fx*dx)
        F_prod = F_prod*FF1

    #Initial value \epsilon_0
    eps_0 = 0

    exp_e = 1-np.exp(eps_0-x)
    # first jj for which 1-exp(eps_0-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+eps_0)/(2*L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(eps_0) and \delta'(eps_0)
    exp_e = 1-np.exp(eps_0-x)
    dexp_e = -np.exp(eps_0-x)
    integrand = exp_e*cfx
    integrand2 = dexp_e*cfx
    sum_int=np.sum(integrand[jj-1:])
    sum_int2=np.sum(integrand2[jj-1:])
    delta_temp = sum_int*dx
    derivative = sum_int2*dx

    # Here tol is the stopping criterion for Newton's iteration
    # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)
    while np.abs(delta_temp - target_delta) > tol_newton:

        #print('Residual of the Newton iteration: ' + str(np.abs(delta_temp - target_delta)))

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta)/derivative

        if(eps_0<-L or eps_0>L):
            break

        # Integrands and integral domain
        exp_e = 1-np.exp(eps_0-x)
        dexp_e = -np.exp(eps_0-x)
        # first kk for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        kk = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))

        integrand = exp_e*cfx
        sum_int=np.sum(integrand[kk-1:])
        delta_temp = sum_int*dx

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e*cfx
        integrand2 = dexp_e*cfx
        sum_int=np.sum(integrand[kk-1:])
        sum_int2=np.sum(integrand2[kk-1:])
        delta_temp = sum_int*dx
        derivative = sum_int2*dx

    if(np.real(eps_0) < -L or np.real(eps_0) > L):
        print('Error: epsilon out of [-L,L] window, please check the parameters.')
        return float('inf')
    else:
        print('Unbounded DP-epsilon after ' + str(int(ncomp)) + ' compositions defined by sigma and q arrays: ' + str(np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
        return np.real(eps_0)








# Parameters:
# target_delta - target delta
# sigma_t - array of sigma values
# q_t - array of q values
# nx - number of points in the discretisation grid
# L -  limit for the integral

def get_eps_bounded(sigma_t,q_t,target_delta=1e-6,nx=1E6,L=20.0):

    nx = int(nx)

    tol_newton = 1e-10 # set this to, e.g., 0.01*target_delta

    dx = 2.0*L/nx # discretisation interval \Delta x
    x = np.linspace(-L,L-dx,nx,dtype=np.complex128) # grid for the numerical integration

    #Initial value \epsilon_0
    eps_0 = 0

    fx_table=[]
    F_prod=np.ones(x.size)

    ncomp=sigma_t.size

    if(q_t.size != ncomp):
        print('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):

        sigma=sigma_t[ij]
        q=q_t[ij]

        ii = 1
        # Evaluate the PLD distribution,
        # This is the case of substitution relation (subsection 5.2)
        ey = np.exp(x[ii-1:])
        c = q*np.exp(-1/(2*sigma**2))
        term1=(-(1-q)*(1-ey) +  np.sqrt((1-q)**2*(1-ey)**2 + 4*c**2*ey))/(2*c)
        term1=np.maximum(term1,1e-16)
        Linvx = (sigma**2)*np.log(term1)

        sq = np.sqrt((1-q)**2*(1-ey)**2 + 4*c**2*ey)
        nom1 = 4*c**2*ey - 2*(1-q)**2*ey*(1-ey)
        term1 = nom1/(2*sq)
        nom2 = term1 + (1-q)*ey
        nom2 = nom2*(sq+(1-q)*(1-ey))
        dLinvx = sigma**2*nom2/(4*c**2*ey)


        ALinvx = (1/np.sqrt(2*np.pi*sigma**2))*((1-q)*np.exp(-Linvx*Linvx/(2*sigma**2))
            + q*np.exp(-(Linvx-1)*(Linvx-1)/(2*sigma**2)))
        fx = np.zeros(nx)
        fx[ii-1:] =  np.real(ALinvx*dLinvx)
        half = int(nx/2)

        # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
        temp = np.copy(fx[half:])
        fx[half:] = np.copy(fx[:half])
        fx[:half] = temp

        FF1 = np.fft.fft(fx*dx) # Compute the DFFT
        F_prod = F_prod*FF1

    exp_e = 1-np.exp(eps_0-x)

    # first jj for which 1-exp(eps_0-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod/dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(eps_0) and \delta'(eps_0)
    exp_e = 1-np.exp(eps_0-x)
    dexp_e = -np.exp(eps_0-x)
    integrand = exp_e*cfx
    integrand2 = dexp_e*cfx
    sum_int=np.sum(integrand[jj-1:])
    sum_int2=np.sum(integrand2[jj-1:])
    delta_temp = sum_int*dx
    derivative = sum_int2*dx

    # Here tol is the stopping criterion for Newton's iteration
    # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)
    while np.abs(delta_temp - target_delta) > tol_newton:

        # print('Residual of the Newton iteration: ' + str(np.abs(delta_temp - target_delta)))

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta)/derivative

        if(eps_0<-L or eps_0>L):
            break

        # Integrands and integral domain
        exp_e = 1-np.exp(eps_0-x)
        dexp_e = -np.exp(eps_0-x)

        # first kk for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        kk = int(np.floor(float(nx*(L+np.real(eps_0))/(2*L))))

        integrand = exp_e*cfx
        sum_int=np.sum(integrand[kk-1:])
        delta_temp = sum_int*dx

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e*cfx
        integrand2 = dexp_e*cfx
        sum_int=np.sum(integrand[kk-1:])
        sum_int2=np.sum(integrand2[kk-1:])
        delta_temp = sum_int*dx
        derivative = sum_int2*dx

    if(np.real(eps_0) < -L or np.real(eps_0) > L):
        print('Error: epsilon out of [-L,L] window, please check the parameters.')
        return float('inf')
    else:
        print('Bounded DP-epsilon after ' + str(int(ncomp)) + ' compositions defined by sigma and q arrays: ' + str(np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
        return np.real(eps_0)
    #
    # print('Bounded DP-epsilon after ' + str(int(ncomp)) + ' compositions:' + str(np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
    # return np.real(eps_0)

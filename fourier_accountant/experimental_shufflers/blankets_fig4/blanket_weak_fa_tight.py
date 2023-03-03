



import numpy as np
from compute_epsilon_bin_weak_tight import get_epsilons,get_omega

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# parameters
n = 1000
# numbers of compositions
ncs=[1,4,16]
k=4
gamma=0.25

# parameters for FA
nx = int(1E7)
L = 100

omega_y,grid_x, error_term = get_omega(n, gamma,k, nx=nx, L=L)

for n_c in ncs:

    half = int(nx/2)
    fx=np.copy(omega_y)
    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp
    # Compute the DFT
    FF1 = np.fft.fft(fx)
    y = np.fft.ifft((FF1**n_c))
    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(y[half:])
    y[half:] = y[:half]
    y[:half] = temp
    print('sum_y: ' + str(sum(y)))




    deltas=np.linspace(-4,-7,5)
    deltas=10**deltas
    epsilons_pld=[]

    for delta in deltas:
        target_delta = delta
        eps = get_epsilons(y,error_term,grid_x,L,nx, target_delta=target_delta)
        epsilons_pld.append(eps)

    print(epsilons_pld)
    pickle.dump(epsilons_pld, open('./pickles/eps_blanket_weak' + str(n_c) + '.p', "wb"))

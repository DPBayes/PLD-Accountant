



import numpy as np
from compute_epsilon_bin_tight import get_epsilons,get_omega

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle


# parameters
n = 1000
# numbers of compositions
#ncs=[1,4,16]
# parameters for PLD accountant
nx = int(3E7)
L = 10

ncs=[1,4,16]
k=4
gamma=0.25
p=gamma/k

omega_y,grid_x,error_term = get_omega(n,gamma,k,nx,L)

for n_c in ncs:

    print(n_c)

    deltas=np.linspace(-4,-7,5)
    deltas=10**deltas
    epsilons_pld=[]

    for delta in deltas:
        #print(delta)
        target_delta = delta
        eps = get_epsilons(omega_y,error_term,grid_x,n=n, gamma=gamma, k=k, n_c=n_c, nx=nx, L=L, target_delta=target_delta)
        #eps = get_epsilons(n=n-1, p=p, n_c=n_c, nx=nx, L=L, target_delta=target_delta)
        epsilons_pld.append(eps)

    print(epsilons_pld)

    pickle.dump(epsilons_pld, open('./pickles/eps_blanket_strong' + str(n_c) + '.p', "wb"))




import numpy as np
import compute_eps
import compute_eps_var

import compute_delta
import compute_delta_var



# Examples for fixed q and sigma

q=0.01
sigma=1.5
nc=1000 #number of compositions
L=20
nx=2E6



# Examples for computing delta as a function of epsilon

eps=1.0 #target epsilon

#Compute delta in remove/add relation
d1  = compute_delta.get_delta_R(q=q,sigma=sigma,target_eps=eps,L=L,nx=nx,ncomp=nc)

#Compute delta in substitute relation
d2  = compute_delta.get_delta_S(q=q,sigma=sigma,target_eps=eps,L=L,nx=nx,ncomp=nc)



# Examples for computing epsilon as a function of delta

delta=1e-7

#Compute epsilon in remove/add relation
e1  = compute_eps.get_epsilon_R(q=q,sigma=sigma,target_delta=delta,L=L,nx=nx,ncomp=nc)

#Compute epsilon in substitute relation
e2  = compute_eps.get_epsilon_S(q=q,sigma=sigma,target_delta=delta,L=L,nx=nx,ncomp=nc)



# Examples for varying sigma and q

L=20
nx=2E6
nc=20
sigmas=np.linspace(1.2,2.2,nc)
q_values=np.linspace(0.05,0.07,nc)

eps=1.5

d1  = compute_delta_var.get_delta_R(q_t=q_values,sigma_t=sigmas,target_eps=eps,L=L,nx=nx)
d2  = compute_delta_var.get_delta_S(q_t=q_values,sigma_t=sigmas,target_eps=eps,L=L,nx=nx)

delta=1e-6

e1 = compute_eps_var.get_eps_R(q_t=q_values,sigma_t=sigmas,target_delta=delta,L=L,nx=nx)
e2 = compute_eps_var.get_eps_S(q_t=q_values,sigma_t=sigmas,target_delta=delta,L=L,nx=nx)



# Test the convergence of the method w.r.t. the parameter n
# nxs=[5E5, 1E6, 2E6, 4E6, 8E6, 1.6E7, 3.2E7]
# deltas = []
# for nx in nxs:
#     c  = compute_delta.get_delta_S(q=q,sigma=sigma,target_eps=eps,L=L,nx=nx,ncomp=nc)
#     deltas.append(c)
# print(deltas)

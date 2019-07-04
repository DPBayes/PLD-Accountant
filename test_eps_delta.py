


import numpy as np
import compute_eps
import compute_eps_var

import compute_delta
import compute_delta_var

#import pld_accountant




# Examples for fixed q and sigma

L=60
nx=2E6
q=0.01
sigma=1.2
nc=10000 #number of compositions

delta=1e-5

a  = compute_eps.get_epsilon_bounded(q=q,sigma=sigma,target_delta=delta,L=L,nx=nx,ncomp=nc)
b  = compute_eps.get_epsilon_unbounded(q=q,sigma=sigma,target_delta=delta,L=L,nx=nx,ncomp=nc)

eps=2.0

c  = compute_delta.get_delta_unbounded(q=q,sigma=sigma,target_eps=eps,L=L,nx=nx,ncomp=nc)
d  = compute_delta.get_delta_bounded(q=q,sigma=sigma,target_eps=eps,L=L,nx=nx,ncomp=nc)



# Examples for varying sigma and q

L=20
nx=2E5
nc=500
sigmas=np.linspace(1.2,1.6,nc)
q_values=np.linspace(0.02,0.03,nc)

delta=1e-5

a = compute_eps_var.get_eps_bounded(q_t=q_values,sigma_t=sigmas,target_delta=delta,L=L,nx=nx)
b = compute_eps_var.get_eps_unbounded(q_t=q_values,sigma_t=sigmas,target_delta=delta,L=L,nx=nx)

eps=1.5

c  = compute_delta_var.get_delta_bounded(q_t=q_values,sigma_t=sigmas,target_eps=eps,L=L,nx=nx)
d  = compute_delta_var.get_delta_unbounded(q_t=q_values,sigma_t=sigmas,target_eps=eps,L=L,nx=nx)

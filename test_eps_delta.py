
'''
A code for computing exact DP guarantees.
The method is described in
A.Koskela, J.Jälkö and A.Honkela:
Computing Exact Guarantees for Differential Privacy.
arXiv preprint arXiv:?.?. (2019)
The code is due to Antti Koskela (@koskeant) and Joonas Jälkö (@jjalko)
'''



import numpy as np
import compute_eps
import compute_delta

L=20
nx=4E6
q=0.01
sigma=2.0
delta=1e-5
nc=50000 #number of compositions

a  = compute_eps.get_epsilon_unbounded(q=q,sigma=sigma,target_delta=delta,L=L,nx=nx,ncomp=nc)

b  = compute_eps.get_epsilon_bounded(q=q,sigma=sigma,target_delta=delta,L=L,nx=nx,ncomp=nc)

# L=20
# nx=2E6
# q=0.01
# sigma=1.0
eps=1.0
# nc=1000 #number of compositions

a  = compute_delta.get_delta_unbounded(q=q,sigma=sigma,target_eps=eps,L=L,nx=nx,ncomp=nc)

b  = compute_delta.get_delta_bounded(q=q,sigma=sigma,target_eps=2*eps,L=L,nx=nx,ncomp=nc)

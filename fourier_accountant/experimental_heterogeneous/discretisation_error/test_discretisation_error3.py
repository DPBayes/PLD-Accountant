
from compute_delta3 import get_deltas
import numpy as np

# parameters for binomial mechanism
n1 = 1000
p  = 0.5

D2 = 1 # squared L2-norm of vector \Delta 

n_comp = 20 # number of compositions

# parameters for PLD accountant
nxs = [1E4,1E5,1E6,1E7,1E8]
L = 5
target_eps=1.0

for nx2 in nxs:
    nx=int(nx2)
    deltas = get_deltas(n1=n1, p=p, D2=D2, n_comp=n_comp,  nx=nx, L=L, target_eps=target_eps)

print(deltas)

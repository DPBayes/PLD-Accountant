
from compute_delta2 import get_deltas
import numpy as np

# parameters for binomial mechanism
n1 = 1000
p  = 0.5

D2 = 1 # squared L2-norm of vector \Delta

n_comp = 20 # number of compositions

# parameters for PLD accountant
nx = int(1E7)
L = 5

target_epsilons=np.linspace(0.3,1.9,5)
for target_eps in target_epsilons:
    deltas = get_deltas(n1=n1, p=p, D2=D2, n_comp=n_comp,  nx=nx, L=L, target_eps=target_eps)


print(deltas)

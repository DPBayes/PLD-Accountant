
from compute_delta import get_deltas

# parameters for binomial mechanism
n1 = 1000
p  = 0.5

D2 = 1 # squared L2-norm of vector \Delta 

n_comp = 28 # number of compositions

# parameters for PLD accountant
nx = int(1E6)
L = 2

target_eps = 1.0

deltas = get_deltas(n1=n1, p=p, D2=D2, n_comp=n_comp,  nx=nx, L=L, target_eps=target_eps)
print(deltas)



import numpy as np

import time
import pickle
from compute_delta import get_delta_max


# Params for PLD
L=10

# Delta
sigma=5.0
eps=2.0
nc=10
nx=1E6
p=0.52

dd_max = get_delta_max(target_eps=eps,sigma=sigma,ncomp=nc,nx=nx,L=L,p=p)



from tensorflow_privacy.privacy.analysis import rdp_accountant

def TF_MA(sigma, T, target_delta=None, target_epsilon=None, max_order=32):
    orders = range(2, max_order + 1)
    rdp = np.zeros_like(orders, dtype=float)
    #print(rdp)
    #print(size(rdp))
    for i in orders:
        # RDP for the Gaussian mechanism
        rdp[i-2] += (T/2)*i/(2*sigma**2)
        # RDP for the randomised response
        rdp[i-2] += (T/2)*(1/(i-1))*np.log( (p**i)*(1-p)**(1-i) + (1-p)**i*p**(1-i)  )
    eps, delta, opt_order = rdp_accountant.get_privacy_spent(orders, rdp, target_delta=target_delta, target_eps=target_epsilon)
    return (eps, delta)

### For target_epsilon = 1.0
target_epsilon=eps
dp_sigma=sigma
a,b = TF_MA(dp_sigma, nc, target_epsilon = target_epsilon)
print('TF: ' + str(a) + ' ' + str(b))

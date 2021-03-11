

import numpy as np

import time
import pickle
from compute_delta import get_delta_max
from tensorflow_privacy.privacy.analysis import rdp_accountant


# Params for PLD
L=10

# number of compositions
#ncs = np.linspace(200,1000,9)

# Delta
sigma=5.0
eps=2.0
nc=30
p=0.52

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


nxs=[5E3,1E4,5E4,1E5]
tf_ts=[]
pld_ts=[]
pld_deltas=[]
tf_deltas=[]

for nx in nxs:
    start_pld = time.process_time()
    dd_max = get_delta_max(target_eps=eps,sigma=sigma,ncomp=nc,nx=nx,L=L,p=p)
    end_pld = time.process_time()
    pld_deltas.append(dd_max)
    pld_t=round(end_pld-start_pld,4)
    pld_ts.append(pld_t)

### For target_epsilon = 1.0
target_epsilon=eps
dp_sigma=sigma
start_tf = time.process_time()
a,b = TF_MA(dp_sigma, nc, target_epsilon = target_epsilon)
end_tf = time.process_time()
tf_t = round(end_tf-start_tf,4)
print('TF: ' + str(a) + ' ' + str(b))

print('tf_t : ' + str(tf_t))
print(pld_ts)

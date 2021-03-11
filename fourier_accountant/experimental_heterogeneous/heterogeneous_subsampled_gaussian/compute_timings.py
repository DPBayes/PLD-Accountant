

import numpy as np

import time
import pickle
from compute_delta import get_delta_max
from tensorflow_privacy.privacy.analysis import rdp_accountant


# Params for PLD
L=10
# subsampling probability
q = 0.01

# Delta
eps=1.0

deltas_max=[]

sigmas=np.linspace(3.0,2.5,11)
nc=500



# From Tensorflow MA: 'compute_heterogenous_rdp'
def compute_heterogenous_rdp(sampling_probabilities, noise_multipliers,
                             steps_list, orders):
  assert len(sampling_probabilities) == len(noise_multipliers)
  rdp = 0
  for q, noise_multiplier, steps in zip(sampling_probabilities,
                                        noise_multipliers, steps_list):
    rdp += rdp_accountant.compute_rdp(q, noise_multiplier, steps, orders)
  return rdp

def TF_MA(q, sigmas, nc, target_delta=None, target_epsilon=None, max_order=32):
    sp = q*np.ones(len(sigmas))
    steps_list = nc*np.ones(len(sigmas))
    orders = range(2, max_order + 1)
    rdp = np.zeros_like(orders, dtype=float)
    rdp += compute_heterogenous_rdp(sp, sigmas, steps_list, orders)
    eps, delta, opt_order = rdp_accountant.get_privacy_spent(orders, rdp, target_delta=target_delta, target_eps=target_epsilon)
    return (eps, delta)

start_tf=0.0
end_tf=0.0
start_pld=0.0
end_pld=0.0

nxs=[5E4,1E5,5E5,1E6]
tf_ts=[]
pld_ts=[]
pld_deltas=[]
tf_deltas=[]
for nx in nxs:
    print('nx : ' + str(nx))
    deltas_pld = []
    deltas_tf = []
    # The main loop
    for ii in range(1,len(sigmas)+1):
        start_pld = time.process_time()
        d1 = get_delta_max(target_eps=eps,sigmas=sigmas[:ii],q=q,ncomp=nc,nx=nx,L=L)
        end_pld = time.process_time()
        deltas_pld.append(d1)
        #print('d1: ' + str(d1))
        start_tf = time.process_time()
        a,d2 = TF_MA(q, sigmas[:ii], nc, target_epsilon = eps)
        end_tf = time.process_time()
        deltas_tf.append(d2)
        #print('d2: ' + str(d2))
    print(d1)
    tf_t=round(end_tf-start_tf,4)
    pld_t=round(end_pld-start_pld,4)
    print('tf_t : ' + str(tf_t))
    print('pld_t : ' + str(pld_t))
    tf_ts.append(tf_t)
    pld_ts.append(pld_t)
    pld_deltas.append(deltas_pld[-1])
    tf_deltas.append(deltas_tf[-1])

print(tf_deltas)
print(pld_deltas)
print(tf_ts)
print(pld_ts)


print( ('%1d' % nxs[0]) + '  &  ' + ('%1d' % nxs[1]) + '  &  ' + ('%1d' % nxs[2]) + '  &  ' +  ('%1d' % nxs[3]) + '\\\\')
print( ('%10.2E' % pld_deltas[0]) + '  &  ' + ('%10.2E' % pld_deltas[1]) + '  &  ' + ('%10.2E' % pld_deltas[2]) + '  &  ' +  ('%10.2E' % pld_deltas[3]) + '\\\\')
print( ('%10.2E' % pld_ts[0]) + '  &  ' + ('%10.2E' % pld_ts[1]) + '  &  ' + ('%10.2E' % pld_ts[2]) + '  &  ' +  ('%10.2E' % pld_ts[3]) + '\\\\')


for (i,nx) in enumerate(nxs):
    print( ('%1d' % nx) + '  &  ' + ('%10.2E' % pld_ts[i]) + '  &  ' + ('%10.5E' % pld_deltas[i]) + '\\\\')

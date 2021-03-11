

import numpy as np

import time
import pickle
from compute_delta import get_delta_max
from compute_delta2 import get_delta_max2
from tensorflow_privacy.privacy.analysis import rdp_accountant


# Params for PLD
L=12
# subsampling probability
q = 0.02

# Delta
eps=1.0

nxs=[5E4,1E5,1E6,5E6]
deltas_max=[]

nc=500

ss=2.0
deltas=[]
time_slow=[]
time_fast=[]

for nx in nxs:
    d1,t1 = get_delta_max(target_eps=eps,sigma=ss,q=q,ncomp=nc,nx=nx,L=L)
    d2,t2 = get_delta_max2(target_eps=eps,sigma=ss,q=q,ncomp=nc,nx=nx,L=L)
    deltas.append(d2)
    time_slow.append(t1)
    time_fast.append(t2)

for (i,nx) in enumerate(nxs):
    t1=time_slow[i]/0.001
    t2=time_fast[i]/0.001
    dd=deltas[i]
    print( ('%1d' % nx) + '  &  ' + ('%10.1E' % t1) + ' & ' + ('%10.1E' % t2) + ' & ' + ('%10.6E' % dd) + '\\\\')

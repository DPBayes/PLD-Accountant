
import numpy as np
import adaptive_discrete_DP

# Parameters for the exponential mechanism
m=50
n=100
eps_em=0.1

#Determine the neghbouring distributions for the exponential mechanism
P1=np.array([np.exp(eps_em*m),  np.exp(eps_em*(n-m))])
P1=P1/np.sum(P1)

P2=np.array([np.exp(eps_em*(m-1)),  np.exp(eps_em*(n-m))])
P2=P2/np.sum(P2)


ncomp=100
nx=int(1E6)
target_eps=2.0

ub = adaptive_discrete_DP.get_delta_upper(P1,P2,target_eps,ncomp,nx)
lb = adaptive_discrete_DP.get_delta_lower(P1,P2,target_eps,ncomp,nx)

print('Upper bound for delta after ' + str(int(ncomp)) + ' compositions: ' + str(ub))
print('Lower bound for delta after ' + str(int(ncomp)) + ' compositions: ' + str(lb))







# Parameter for the randomised response
p=0.4

#Determine the neghbouring distributions for the randomised response
P1=np.array([1-p,  p])
P1=P1/np.sum(P1)

P2=np.array([p,  1-p])
P2=P2/np.sum(P2)

ncomp=10
nx=int(1E6)
target_eps=4.0

ub = adaptive_discrete_DP.get_delta_upper(P1,P2,target_eps,ncomp,nx)
lb = adaptive_discrete_DP.get_delta_lower(P1,P2,target_eps,ncomp,nx)

print('Upper bound for delta after ' + str(int(ncomp)) + ' compositions: ' + str(ub))
print('Lower bound for delta after ' + str(int(ncomp)) + ' compositions: ' + str(lb))

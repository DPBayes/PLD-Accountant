



import numpy as np
#from compute_epsilon_bin import get_epsilons

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm

import pickle



deltas=np.linspace(-4,-7,5)
deltas=10**deltas

eps_blankets=[]
eps_blankets_tight=[]
eps_blankets_tight2=[]


# parameters
n = 500
# numbers of compositions
ncs=[1,4,16]
k=4
gamma=0.25
p=gamma/k


for nc2 in ncs:
    nc=int(nc2)
    eps_blanket = pickle.load(open('./pickles/eps_blanket_strong' + str(nc) + '.p', "rb"))
    eps_blankets.append(eps_blanket)

for nc2 in ncs:
    nc=int(nc2)
    eps_blanket_tight = pickle.load(open('./pickles/eps_blanket_feldman_fa' + str(nc) + '.p', "rb"))
    eps_blankets_tight.append(eps_blanket_tight)

for nc2 in ncs:
    nc=int(nc2)
    eps_blanket_tight2 = pickle.load(open('./pickles/eps_blanket_weak' + str(nc) + '.p', "rb"))
    eps_blankets_tight2.append(eps_blanket_tight2)




pp = PdfPages('./plots/blankets_feldman_fa.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 12.0})
plt.rc('text', usetex=True)
plt.rc('font', family='arial')

plt.xlabel(r'$\delta$')
plt.ylabel(r'$\varepsilon$')
plt.ylim((0,5.0))
plt.xlim((8e-8,1.5e-4))

Ncols = 9

colors = [matplotlib.cm.viridis(x) for x in np.linspace(0, 0.9, Ncols)]


plt.semilogx(deltas,eps_blankets_tight[2],'-.',color=colors[4])
plt.semilogx(deltas,eps_blankets_tight[1],'--',color=colors[4])
plt.semilogx(deltas,eps_blankets_tight[0],'-',color=colors[4])

#for eps_blanket in eps_blankets:
plt.semilogx(deltas,eps_blankets[2],'-.',color=colors[0])
plt.semilogx(deltas,eps_blankets[1],'--',color=colors[0])
plt.semilogx(deltas,eps_blankets[0],'-',color=colors[0])

plt.semilogx(deltas,eps_blankets_tight2[2],'-.',color=colors[8])
plt.semilogx(deltas,eps_blankets_tight2[1],'--',color=colors[8])
plt.semilogx(deltas,eps_blankets_tight2[0],'-',color=colors[8])




legs=[]


for nc2 in ncs[::-1]:
    nc=int(nc2)
    legs.append('Feldman et al. (2023, Thm. 5.1), $n_c=' + str(nc) + '$')

for nc2 in ncs[::-1]:
    nc=int(nc2)
    legs.append('Adversary $A_s$, $n_c=' + str(nc) + '$')

for nc2 in ncs[::-1]:
     nc=int(nc2)
     legs.append('Adversary $A_w$, $n_c=' + str(nc) + '$')



plt.legend(legs,loc='upper right')
pp.savefig(plot_, bbox_inches = 'tight', pad_inches = 0)
pp.close()


plt.show()
plt.close()

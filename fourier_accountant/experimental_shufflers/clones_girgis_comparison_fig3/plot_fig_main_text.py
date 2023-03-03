



import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm

import pickle



deltas=np.linspace(-3,-7,5)
deltas=10**deltas

eps_blankets=[]
eps_blankets_tight=[]


# parameters
n = 100
# numbers of compositions
ncs=[1,4,16]
k=4
gamma=0.25
p=gamma/k

eps_girgis_upper = pickle.load(open('./pickles/eps_fig1_girgis_upper.p', 'rb'))

eps_pld = pickle.load(open('./pickles/eps_fig1_pld.p', 'rb'))
eps_pld_addremove = pickle.load(open('./pickles/eps_fig1_addremove_pld.p', 'rb'))

print(len(eps_girgis_upper))
print(len(eps_pld))


Ts =  [10,int(10**1.5),10**2,int(10**2.5),10**3,int(10**3.5),10**4]


pp = PdfPages('./plots/girgis_fig0.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 12.0})
plt.rc('text', usetex=True)
plt.rc('font', family='arial')

plt.xlabel('Number of compositions $n_c$')
plt.ylabel(r'DP $\varepsilon$')


Ncols = 9

colors = [matplotlib.cm.viridis(x) for x in np.linspace(0, 0.8, Ncols)]

#for eps_blanket in eps_blankets:
lw=1.0
plt.loglog(Ts,eps_girgis_upper,'-o',color=colors[0],linewidth=lw)

plt.loglog(Ts,eps_pld,'-d',color=colors[5],linewidth=lw)
#plt.loglog(Ts,eps_pld_addremove,'-x',color=colors[8],linewidth=lw)

#matplotlib.pyplot
plt.grid(True)
plt.title(r'$\varepsilon_0=3.0$, $n=10^5$, $q=0.01$, $\delta=1/n$')
legs=[]

legs.append(r'$\varepsilon$ from RDP upper bound (Girgis et al., 2021)')

legs.append(r'$\varepsilon$ from PLD + FFT accountant')
#legs.append('$(P,Q)$ from Feldman et al. (2023), add/remove relation')

plt.legend(legs,loc='lower right')
pp.savefig(plot_, bbox_inches = 'tight', pad_inches = 0)
pp.close()


plt.show()
plt.close()

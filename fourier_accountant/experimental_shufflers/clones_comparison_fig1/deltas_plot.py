



import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import pickle

n_eps=60
epsilons = np.linspace(0.1,1.0,n_eps)

deltas_pld = pickle.load(open("./pickles/deltas_pld.p", "rb"))
deltas_feldman = pickle.load(open("./pickles/deltas_feldman.p", "rb"))
deltas_feldman2 = pickle.load(open("./pickles/deltas_feldman2.p", "rb"))
deltas_feldman3 = pickle.load(open("./pickles/deltas_feldman3.p", "rb"))
deltas_feldman4 = pickle.load(open("./pickles/deltas_feldman4.p", "rb"))


pp = PdfPages('./plots/deltas_compare_k_n1e4.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 15.0})
plt.rc('text', usetex=True)
plt.rc('font', family='arial')

plt.ylabel('$\delta$')
plt.xlabel(r'$\varepsilon$')

Ncols = 8
colors = [matplotlib.cm.viridis(x) for x in np.linspace(0, 0.8, Ncols)]

plt.semilogy(epsilons,deltas_feldman4,'-',color=colors[3])
plt.semilogy(epsilons,deltas_feldman3,'-',color=colors[2])
plt.semilogy(epsilons,deltas_feldman2,'-',color=colors[1])
plt.semilogy(epsilons,deltas_feldman,'-',color=colors[0])


deltasdeltas=[]
for deltas in deltas_pld:
    deltasdeltas.append(deltas)

deltasdeltas.reverse()

for i,deltas in enumerate(deltasdeltas):
    plt.semilogy(epsilons,deltas,'--',color=colors[i+4])

legs=[]

legs.append('RDP, $n_c=4$')
legs.append('RDP, $n_c=3$')
legs.append('RDP, $n_c=2$')
legs.append('RDP, $n_c=1$')

legs.append('PLD, $n_c=4$')
legs.append('PLD, $n_c=3$')
legs.append('PLD, $n_c=2$')
legs.append('PLD, $n_c=1$')

plt.legend(legs,loc='lower left')
plt.ylim([1E-13,1])

pp.savefig(plot_, bbox_inches = 'tight', pad_inches = 0)
pp.close()


plt.show()
plt.close()

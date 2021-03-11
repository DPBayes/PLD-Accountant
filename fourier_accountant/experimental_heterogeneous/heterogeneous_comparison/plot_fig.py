

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# number of compositions
ncs=np.linspace(6,30,13)

deltas_tf = pickle.load(open("./pickles/deltas_tf.p", "rb"))
deltas_pld = pickle.load(open("./pickles/deltas_pld.p", "rb"))


pp = PdfPages('./plots/deltas_RR.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 11.0})

plt.semilogy(ncs,deltas_tf[0],'--',marker='+',linewidth=.75,markersize=7)
plt.semilogy(ncs,deltas_pld[0],'-')

plt.semilogy(ncs,deltas_tf[1],'--',marker='+',linewidth=.75,markersize=7)
plt.semilogy(ncs,deltas_pld[1],'-')

plt.xlabel("Number of compositions $k$")
plt.ylabel("$\delta(\epsilon)$")

legs = []
legs.append('TF MA, $\epsilon=2.0$')
legs.append('FA, $\epsilon=2.0$')

legs.append('TF MA, $\epsilon=4.0$')
legs.append('FA, $\epsilon=4.0$')

plt.legend(legs,loc='lower right')

plt.ylim(1E-14,1)
pp.savefig(plot_, bbox_inches = 'tight', pad_inches = 0)
pp.close()
plt.show()
plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# number of compositions
nc=500
ncd=11
ncs=np.linspace(nc,ncd*nc,ncd)

deltas_tf = pickle.load(open("./pickles/deltas_tf.p", "rb"))
deltas_pld = pickle.load(open("./pickles/deltas_pld.p", "rb"))

deltas_tf2 = pickle.load(open("./pickles/deltas_tf2.p", "rb"))
deltas_pld2 = pickle.load(open("./pickles/deltas_pld2.p", "rb"))


pp = PdfPages('./plots/deltas_gauss.pdf')
plot_ = plt.figure()
plt.rcParams.update({'font.size': 11.0})


plt.semilogy(ncs,deltas_tf2,'--',marker='+',linewidth=.75,markersize=7)
plt.semilogy(ncs,deltas_pld2,'-')

plt.semilogy(ncs,deltas_tf,'--',marker='+',linewidth=.75,markersize=7)
plt.semilogy(ncs,deltas_pld,'-')


plt.xlabel("Number of compositions $k$")
plt.ylabel("$\delta(\epsilon)$")

legs = []
legs.append('TF MA, $q=0.02$, $\sigma = (3.0, \ldots, 2.0)$')
legs.append('FA, $q=0.02$, $\sigma = (3.0, \ldots, 2.0)$')
legs.append('TF MA, $q=0.01$, $\sigma = (3.0, \ldots, 2.5)$')
legs.append('FA, $q=0.01$, $\sigma = (3.0, \ldots, 2.5)$')
plt.legend(legs,loc='lower right')



plt.ylim(1E-12,0.3)
pp.savefig(plot_, bbox_inches = 'tight', pad_inches = 0)
pp.close()
plt.show()
plt.close()

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:58:02 2021

@author: pallotto
"""


from brian2 import *
from brian2tools import *
prefs.codegen.target = 'numpy'
import numpy as np

#STIMULUS DESIGN
list1=np.array(range(0,33))
list2=np.array(range(33, 66))
list3=np.array(range(66,99))

indices=np.concatenate((list1,list2,list3))



# Stim only A (all)
timeA=10
timeB=10
timeC=10

stimA=np.ones(33)*timeA
stimB=np.zeros(33)*timeB
stimC=np.zeros(33)*timeC

times= np.concatenate((stimA,stimB,stimC))*ms




start_scope()

#indices = array([1,1,1,1,1,50,50,50,50,50,19,19,19,19,19])
#times = array([10, 12, 14,40,50,20, 22, 24,40,10,10, 30, 40,50,20])*ms

N=100

tau=10*ms
tau_t=50*ms
f=100*Hz

eqs= ''' 
dv/dt = (1-v)/tau +0.5*xi*tau**-0.5 : 1 (unless refractory)
dvt/dt= (1-vt)/tau_t :1 
I : 1
'''



inp = SpikeGeneratorGroup(N, indices, times)
G = NeuronGroup(N, eqs, threshold='v>2*vt', reset='v = 0; vt+=0.2', 
                method='euler', refractory=1*ms,)
G.v=0.5
w=0
feedforward = Synapses(inp, G, on_pre='v+=2') 
feedforward.connect(j='i')
recurrent = Synapses(G, G, on_pre='v+=w')
recurrent.connect('i!=j')


# Monitor neurons
M = StateMonitor(G, 'v', record=True)
spM = SpikeMonitor(G)


run(100*ms)


figure(figsize=(10,5))
subplot(121)
brian_plot(M)
subplot(122)
plot(spM.t/ms, spM.i, '.')




# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:44:57 2021

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
timeA=50
timeB=30
timeC=40

stimA=np.ones(33)*timeA
stimB=np.ones(33)*timeB
stimC=np.ones(33)*timeC

times= np.concatenate((stimA,stimB,stimC))*ms

start_scope()


N=99

tau=10*ms
tau_t=50*ms
f=100*Hz
sigma = 0.5

eqs= ''' 
dv/dt = (-v+I_ei)/tau +sigma*xi*tau**-0.5 : 1 (unless refractory)
dvt/dt= (1-vt)/tau_t :1 
I : 1
dI_ee/dt = -I_ee/tau : 1
dI_ei/dt = -I_ei/tau : 1 
'''

inp = SpikeGeneratorGroup(N, indices, times)
exc_neurons = NeuronGroup(N, eqs, threshold='v>2*vt', reset='v = 0; vt+=0.2', 
                # Synapses are modeled as exponentially decaying currents
                method='euler', refractory=1*ms,)
exc_neurons.v='rand()*0.5'
w=0.025
feedforward = Synapses(inp, exc_neurons, on_pre='v+=2') 
feedforward.connect(j='i')
E_to_E = Synapses(exc_neurons, exc_neurons, on_pre='v+=w', on_post='v+=w')
#E_to_E.connect('i!=j')
E_to_E.connect(p=1)

# Add ihibitory neurons

sigma=0.5

inh_neurons = NeuronGroup(800, '''dv/dt = (-v + I_ie )/(5*ms) + sigma*xi*tau**-0.5: 1
                                  dI_ie/dt = -I_ie/tau : 1
                                  ''', threshold='v>0.85', reset='v=0',
                         method='euler', refractory=1*ms)
inh_neurons.v = 'rand()*0.5'

# Synaptic connections (all unspecific)

I_to_E = Synapses(inh_neurons, exc_neurons, on_pre='I_ei -= w', on_post= 'I_ei += w')
I_to_E.connect(p=0)
E_to_I = Synapses(exc_neurons, inh_neurons, on_pre='I_ie += w', on_post='I_ie-=w')
E_to_I.connect(p=0)


# Monitor neurons
exc_mon = StateMonitor(exc_neurons, 'v',record= True)
inh_mon = StateMonitor(inh_neurons, 'v', record= True)
spM_e=SpikeMonitor(exc_neurons)
spM_i=SpikeMonitor(inh_neurons)



run(100*ms)


figure(figsize=(15,5))
subplot(131)
brian_plot(exc_mon)
subplot(132)
plot(spM_e.t/ms, spM_e.i, '.')
subplot(133)
plot(spM_i.t/ms, spM_i.i, '.')

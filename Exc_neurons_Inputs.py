# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:44:57 2021

@author: pallotto
"""

from brian2 import *
from brian2tools import *

duration = 100*ms

# Neuron model parameters
vr = -70*mV
vt = -55*mV
taum = 10*ms
taupsp = 0.325*ms
weight = 4.86*mV
# Neuron model
eqs = Equations('''
dv/dt = (-(v-vr)+x)*(1./taum) : volt
dx/dt = (-x+y)*(1./taupsp) : volt
dy/dt = -y*(1./taupsp)+25.27*mV/ms+
        (39.24*mV/ms**0.5)*xi : volt
''')

# Neuron groups
n_groups = 3
group_size = 50
exc_neurons = NeuronGroup(N=n_groups*group_size, model=eqs,
                threshold='v>vt', reset='v=vr', refractory=1*ms,
                method='euler')


N_inputs=50
Exc_input = SpikeGeneratorGroup(N_inputs, np.arange(N_inputs),
                             np.random.randn(N_inputs)*1*ms + 50*ms)
# The network structure
S = Synapses(exc_neurons,exc_neurons, on_pre='y+=weight')
S.connect(j='k for k in range((int(i/group_size)+1)*group_size, (int(i/group_size)+2)*group_size) '
            'if i<N_pre-group_size')
Sinput = Synapses(Exc_input, exc_neurons[:49], on_pre='y+=weight')
Sinput.connect()




# Record the spikes
exc_spikes = SpikeMonitor(exc_neurons)
exc_mon = StateMonitor(exc_neurons, 'v',record= 4)

# Setup the network, and run it
exc_neurons.v= 'vr + rand() * (vt - vr)'
run(duration)

#Plot

figure(figsize=(10,5))
subplot(121)
plot(exc_spikes.t/ms, 1.0*exc_spikes.i/group_size, '.')
plot([0, duration/ms], np.arange(n_groups).repeat(2).reshape(-1, 2).T, 'k-')
ylabel('group number')
yticks(np.arange(n_groups))
xlabel('time (ms)')
subplot(122)
brian_plot(exc_mon)
show()
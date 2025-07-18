
# # Toy model of predictive coding network with controlled oscillators

# Toy model of predictive coding network implemented in Nengo.
# The code realizes "predictive coding light" concept. The error signal teaches the integrator of upper hierarchy and becomes zero when finished teaching. The input signal and the signal from integrator specify the current error.
# 
# Used in eye control?
# Movement pacing

# Version 2.
# Error fixed:
# - Different names are given to different recurrent functions
# - Stabilization of oscillation amplitude is added

# prepare environment with python 3.9
#!pip install nengo
#!pip install nengo-gui

import nengo

model = nengo.Network()

# integrator part is taken from Nengo summer school lecture 4
# integrator equation
# dx/dt = u(t)
# where u(t) is the input stimulus

# Parameters
tau_synapse = 0.2 # should be reasonably large
omega = 10 # oscillation frequency
gamma = 1 # decay/amplification rate

with model:
    # Integrator part
    
    err = nengo.Ensemble(n_neurons=100, dimensions=1)
    layer1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim = nengo.Node(0)
    layer2 = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim,layer1)
    nengo.Connection(layer1, err)
    
    def forward(u):
        return tau_synapse*u
    # feedforward error
    nengo.Connection(err, layer2, function=forward, synapse=tau_synapse)
    
    def recurrent_integ(x):
        return x
    nengo.Connection(layer2, layer2, function=recurrent_integ, synapse=tau_synapse)
    
    nengo.Connection(layer2, err, transform=-1) # feedback to the error population
    
    # Oscillator part
    
    # adding controlled oscillators
    osc1 = nengo.Ensemble(n_neurons=500, dimensions=4, radius=2)
    osc2 = nengo.Ensemble(n_neurons=500, dimensions=4, radius=2)
    
    def recurrent_osc(x):
        return [-tau_synapse*x[2]*omega*x[1]-tau_synapse*gamma*x[3]*x[0]+x[0], 
                 tau_synapse*x[2]*omega*x[0]+x[1]]
    
    nengo.Connection(osc1,osc1[:2], function=recurrent_osc, synapse=tau_synapse)
    nengo.Connection(osc2,osc2[:2], function=recurrent_osc, synapse=tau_synapse)
    nengo.Connection(layer1, osc1[2])   
    nengo.Connection(layer2, osc2[2])      
    
    # adding amplitude stabilization
    amplitude = nengo.Node(1) # desired oscillation amplitude

    # Ensembles that calculate amplitude error
    amp_err1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    amp_err2 = nengo.Ensemble(n_neurons=100, dimensions=1)
        
    def square(x):
        return x*x
    
    def square2(x):
        return x[0]*x[0]+x[1]*x[1]
    
    nengo.Connection(amplitude, amp_err1, function=square, transform=-1)
    nengo.Connection(amplitude, amp_err2, function=square, transform=-1)
    
    nengo.Connection(amp_err1,osc1[3]) # feedforward amplitude control
    nengo.Connection(amp_err2,osc2[3]) # feedforward amplitude control
    
    # feedback connections to error
    nengo.Connection(osc1, amp_err1, function=square2) 
    nengo.Connection(osc2, amp_err2, function=square2)     
    
    
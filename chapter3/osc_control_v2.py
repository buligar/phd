import nengo
import numpy as np
import matplotlib.pyplot as plt
model = nengo.Network()


tau_synapse = 0.1
omega = 10

with model:
    speed = nengo.Ensemble(n_neurons=50, dimensions=1)
    stim_speed = nengo.Node(1)
    nengo.Connection(stim_speed, speed)
    osc = nengo.Ensemble(n_neurons=500, dimensions=3, radius=2)
    osc2 = nengo.Ensemble(n_neurons=500, dimensions=3, radius=2)
    err = nengo.Ensemble(n_neurons=500, dimensions=3, radius=2)

    def recurrent(x):
        return [-tau_synapse*x[2]*omega*x[1]+x[0], tau_synapse*x[2]*omega*x[0]+x[1]]
    
    nengo.Connection(osc, osc[:2], function=recurrent, synapse=tau_synapse)
    
    def stim_func(t):
        if t < 0.1:
            return 1, 0
        else:
            return 0, 0
        
    stim = nengo.Node(stim_func)
    nengo.Connection(stim, osc[:2])
    nengo.Connection(speed, osc[2]) 
    
    nengo.Connection(osc2, osc2[:2], function=recurrent, synapse=tau_synapse)
    nengo.Connection(osc, err, transform=-1)
    nengo.Connection(osc2, err)
    a = nengo.Connection(err, osc2, learning_rule_type=nengo.PES())
    nengo.Connection(err, a.learning_rule)

    p_osc  = nengo.Probe(osc,  synapse=0.1)
    p_osc2 = nengo.Probe(osc2, synapse=0.1)
    p_err  = nengo.Probe(err,  synapse=0.1)

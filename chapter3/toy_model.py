import nengo
from nengo.processes import Piecewise
import time
start_time = time.time()
model = nengo.Network()

# Parameters
tau_synapse = 0.2

with model:
    err = nengo.Ensemble(n_neurons=100, dimensions=1, radius=1.1)
    layer1 = nengo.Ensemble(n_neurons=100, dimensions=1, radius=1.1)
    stim = nengo.Node(Piecewise({0: 0, 0.2: 1, 3: -1, 10: 0.5}))
    layer2 = nengo.Ensemble(n_neurons=100, dimensions=1, radius=1.1)
    nengo.Connection(stim,layer1)
    nengo.Connection(layer1, err)
    
    def forward(u):
        return tau_synapse*u
    # feedforward error
    nengo.Connection(err, layer2, function=forward, synapse=tau_synapse)
    
    def recurrent(x):
        return x
    nengo.Connection(layer2, layer2, function=recurrent, synapse=tau_synapse)
    nengo.Connection(layer2, err, transform=-1) # feedback to the error population

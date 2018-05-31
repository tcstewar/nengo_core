import numpy as np
import nengo

# generate nengo model just to grab network values from

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=2,
                         neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
                         )
    # change this to change the function approximated by the neurons initially
    def my_func(x):
        return [0, 0]
    # set size_in to be the dimensionality of the result from my_func
    output = nengo.Node(None, size_in=2)
    conn = nengo.Connection(ens, output, function=my_func,
                            synapse=0.01,
                            learning_rule_type=nengo.PES(learning_rate=1e-4,
                                                    pre_tau=0.01))

    error = nengo.Node(None, size_in=2)
    nengo.Connection(output, error, synapse=None)

    stim = nengo.Node(lambda t: [np.sin(t*2*np.pi), np.cos(t*2*np.pi)])
    nengo.Connection(stim, ens, synapse=None)
    nengo.Connection(stim, error, synapse=None, transform=-1)

    nengo.Connection(error, conn.learning_rule, synapse=None)

    p_stim = nengo.Probe(stim)
    p_error = nengo.Probe(error)
    p_output = nengo.Probe(output)
sim = nengo.Simulator(model, dt=0.001)

T = 10.0               # amount of time to run for
sim.run(T)

import pylab
pylab.plot(sim.trange(), sim.data[p_output])
pylab.plot(sim.trange(), sim.data[p_stim], lw=2)
#pylab.plot(t, data_error)
#pylab.xlim(T-1.0, T)
pylab.show()

import numpy as np
import nengo

# generate nengo model just to grab network values from

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=2,
                         neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
                         )
    # change this to change the function approximated by the neurons
    def my_func(x):
        return x
    # set size_in to be the dimensionality of the result from my_func
    output = nengo.Node(None, size_in=2)
    conn = nengo.Connection(ens, output, function=my_func)
sim = nengo.Simulator(model, dt=0.001, progress_bar=False)


max_neurons_per_core = 10


# now use the nengo model to create our own implementation
import core

cores = []
start = 0
while start < ens.n_neurons:
    n_neurons = np.minimum(max_neurons_per_core, ens.n_neurons-start)
    c = core.Core(n_inputs=ens.dimensions,
                  n_neurons=n_neurons,
                  n_outputs=output.size_in,
                  encoders=sim.data[ens].scaled_encoders[start:start+n_neurons],
                  bias=sim.data[ens].bias[start:start+n_neurons],
                  decoders=sim.data[conn].weights[:,start:start+n_neurons],
                  tau_rc=ens.neuron_type.tau_rc,
                  tau_ref=ens.neuron_type.tau_ref,
                  dt=sim.dt,
                  learning_rate=0,
                  )
    cores.append(c)
    start += n_neurons

# test the implementation by running it

T = 1.0               # amount of time to run for
steps = int(T / c.dt)

data_stim = []
data_output = []

for i in range(steps):
    # generate some input to give the model
    t = i * c.dt
    stim = [np.sin(t*2*np.pi), np.cos(t*2*np.pi)]
    data_stim.append(stim)

    # run the model one time step
    outputs = [c.step(stim, error=None) for c in cores]
    output = np.sum(outputs, axis=0)
    data_output.append(output)


# do a 30ms lowpass filter to smooth the output for plotting purposes
smoothed_output = nengo.synapses.Lowpass(0.03).filt(data_output, dt=c.dt)

import pylab
pylab.plot(smoothed_output, label='output')
pylab.plot(data_stim, lw=2, label='input')
pylab.legend(loc='lower right')
pylab.savefig('test_split.png')
pylab.show()

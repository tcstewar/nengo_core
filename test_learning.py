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
    conn = nengo.Connection(ens, output, function=my_func)
sim = nengo.Simulator(model, dt=0.001, progress_bar=False)


# now use the nengo model to create our own implementation

import core
c = core.Core(n_inputs=ens.dimensions,
              n_neurons=ens.n_neurons,
              n_outputs=output.size_in,
              encoders = sim.data[ens].scaled_encoders,
              bias = sim.data[ens].bias,
              decoders = sim.data[conn].weights,
              tau_rc=ens.neuron_type.tau_rc,
              tau_ref=ens.neuron_type.tau_ref,
              dt=sim.dt,
              learning_rate=1e-4,
              )


# test the implementation by running it

T = 6.0               # amount of time to run for
steps = int(T / c.dt)

data_stim = []
data_output = []
data_error = []

# how much smoothing to apply when computing the error
filter_time = 0.01   # lowpass filter time constant in seconds
filter_scale = np.exp(-c.dt/filter_time)
prev_output = np.zeros(c.n_outputs)

for i in range(steps):
    # generate some input to give the model
    t = i * c.dt
    stim = [np.sin(t*2*np.pi), np.cos(t*2*np.pi)]
    data_stim.append(stim)

    # this is the function we want it to learn
    def desired_func(x):
        return x
    ideal_output = desired_func(stim)

    # compute the error
    error = prev_output - ideal_output
    data_error.append(error)

    # run the model one time step, feeding in the error
    output = c.step(stim, error)
    data_output.append(output)

    # keep track of previous output value (for computing the error)
    #  (with a lowpass filter to smooth it a bit)
    prev_output = prev_output*(1-filter_scale) + output*filter_scale


# do a 30ms lowpass filter to smooth the output for plotting purposes
smoothed_output = nengo.synapses.Lowpass(0.03).filt(data_output, dt=c.dt)

import pylab
pylab.plot(smoothed_output, label='output')
pylab.plot(data_stim, lw=2, label='input')
pylab.legend(loc='lower right')
pylab.savefig('test_learning.png')
pylab.show()

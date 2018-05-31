import nengo
import nengo.spa as spa
import numpy as np

# Generate a test network

D = 16         # this goes up to 512 in Spaun!
SD = 16
n_per_d = 100
pstc = 0.01
n_cconv = 200
seed = 6
dt = 0.001

model = spa.SPA(seed=seed)
with model:
    model.inA = spa.Buffer(D, subdimensions=SD,
                           neurons_per_dimension=n_per_d)
    model.inB = spa.Buffer(D, subdimensions=SD,
                           neurons_per_dimension=n_per_d)

    model.result = spa.Buffer(D, subdimensions=SD,
                              neurons_per_dimension=n_per_d)

    model.cortical = spa.Cortical(spa.Actions('result = inA * inB'),
                                  synapse=pstc,
                                  neurons_cconv=n_cconv)

    input_A = nengo.Node(None, size_in=D)
    input_B = nengo.Node(None, size_in=D)
    output = nengo.Node(None, size_in=D)
    nengo.Connection(input_A, model.inA.state.input, synapse=None)
    nengo.Connection(input_B, model.inB.state.input, synapse=None)
    nengo.Connection(model.result.state.output, output, synapse=None)
sim = nengo.Simulator(model, dt=dt)


# convert this model into cores and messages
import system
s = system.System(model, sim)


# Now run the model with a fixed input to evaluate it
T = 0.2
dt = sim.dt
steps = int(T/dt)

# generate the random input
vocab = spa.Vocabulary(D, rng=np.random.RandomState(seed=seed))
A = vocab.parse('A').v
B = vocab.parse('B').v
ideal_result = vocab.parse('A*B').v

data_A = []
data_B = []
data_C = []
for i in range(steps):
    # inject inputs from the outside world
    s.input_values[s.node2inter[input_A]][:] = A
    s.input_values[s.node2inter[input_B]][:] = B

    output_values = s.step()

    #save some data for plotting
    data_A.append(output_values[s.node2inter[model.inA.state.output]].copy())
    data_B.append(output_values[s.node2inter[model.inB.state.output]].copy())
    data_C.append(output_values[s.node2inter[model.result.state.output]].copy())


# plot the results (with a lowpass filter)
filt = nengo.synapses.Lowpass(0.03)

import pylab
pylab.figure(figsize=(14,6))
pylab.subplot(1, 4, 1)
pylab.title('input A')
pylab.plot(filt.filt(data_A, dt=sim.dt, y0=0))
for v in A:
    pylab.axhline(v, ls='--')
pylab.ylim(-1,1)
pylab.subplot(1, 4, 2)
pylab.title('input B')
pylab.plot(filt.filt(data_B, dt=sim.dt, y0=0))
for v in B:
    pylab.axhline(v, ls='--')
pylab.ylim(-1,1)
pylab.subplot(1, 4, 3)
pylab.title('result')
pylab.plot(filt.filt(data_C, dt=sim.dt, y0=0))
for v in ideal_result:
    pylab.axhline(v, ls='--')
pylab.ylim(-1,1)
pylab.subplot(1, 4, 4)
pylab.title('accuracy')
# compute accuracy (normalized dot product with ideal)
filt_result = filt.filt(data_C, dt=sim.dt, y0=0)
accuracy = [np.dot(ideal_result/np.linalg.norm(ideal_result), 
                   x/np.linalg.norm(x)) for x in filt_result]
pylab.plot(accuracy)
pylab.ylim(-1,1)
pylab.savefig('test_connectivity.png')
pylab.show()

import nengo
import nengo.spa as spa
import numpy as np

D = 16
SD = 16
n_per_d = 100
pstc = 0.01
n_cconv = 200
seed = 6

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
sim = nengo.Simulator(model)



import core

conns_out = {}
conns_in = {}
for ens in model.all_ensembles:
    conns_out[ens] = []
    conns_in[ens] = []
for node in model.all_nodes:
    conns_out[node] = []
    conns_in[node] = []
for c in model.all_connections:
    conns_out[c.pre_obj].append(c)
    conns_in[c.post_obj].append(c)

cores = {}
dec_funcs = {}
conn_dec_range = {}
for ens in model.all_ensembles:
    funcs = []
    dec = []
    size = 0
    ranges = []
    for conn in conns_out[ens]:
        if conn.function not in funcs:
            funcs.append(conn.function)
            dec.append(sim.data[conn].weights)
            width = sim.data[conn].weights.shape[0]
            ranges.append((size, size+width))
            size += width
        index = funcs.index(conn.function)
        conn_dec_range[conn] = ranges[index]
    dec = np.vstack(dec)

    c = core.Core(n_inputs=ens.dimensions,
                  n_neurons=ens.n_neurons,
                  n_outputs=dec.shape[0],
                  encoders = sim.data[ens].scaled_encoders,
                  bias = sim.data[ens].bias,
                  decoders = dec,
                  tau_rc=ens.neuron_type.tau_rc,
                  tau_ref=ens.neuron_type.tau_ref,
                  dt=sim.dt,
                  learning_rate=0,
                  )
    cores[ens] = c

input_values = {}
for n in model.all_nodes:
    assert n.output is None
    input_values[n] = np.zeros(n.size_in)
for ens in model.all_ensembles:
    input_values[ens] = np.zeros(ens.dimensions)


class Message(object):
    def __init__(self, 
                 pre_obj, pre_start, pre_len, 
                 post_obj, post_start, post_len,
                 matrix, synapse, dt):
        self.pre_obj = pre_obj
        self.pre_start = pre_start
        self.pre_len = pre_len
        self.post_obj = post_obj
        self.post_start = post_start
        self.post_len = post_len
        self.matrix = matrix
        if synapse is None:
            self.synapse_scale = None
        else:
            self.synapse_scale = np.exp(-dt/synapse)
        self.value = np.zeros(post_len)
    def apply(self, input_values, output_values):
        v = output_values[self.pre_obj][self.pre_start:self.pre_start+self.pre_len]
        v = np.dot(self.matrix, v)
        if self.synapse_scale is not None:
            self.value = (1-self.synapse_scale)*self.value + self.synapse_scale*v
            v = self.value
        input_values[self.post_obj][self.post_start:self.post_start+self.post_len] += v



messages = []
for conn in model.all_connections:
    pre_obj = conn.pre_obj
    if conn.function is None:
        pre_indices = np.arange(pre_obj.size_out)[conn.pre_slice]
    else:
        pre_indices = np.arange(conn.size_mid)

    if isinstance(pre_obj, nengo.Ensemble):
        r = conn_dec_range[conn]
        indices = np.arange(cores[pre_obj].n_outputs)
        indices = indices[r[0]:r[1]]
        pre_indices = indices[pre_indices]

    post_obj = conn.post_obj
    post_indices = np.arange(post_obj.size_in)[conn.post_slice]

    pre_start = pre_indices[0]
    pre_len = pre_indices[-1] - pre_start + 1
    if pre_len != len(pre_indices):
        print(pre_indices)
    assert pre_len == len(pre_indices)

    post_start = post_indices[0]
    post_len = post_indices[-1] - post_start + 1
    assert post_len == len(post_indices)

    m = Message(pre_obj, pre_start, pre_len,
                post_obj, post_start, post_len,
                matrix=conn.transform,
                synapse=conn.synapse if conn.synapse is None else conn.synapse.tau,
                dt=sim.dt)
    messages.append(m)




T = 0.2
dt = sim.dt
steps = int(T/dt)

vocab = spa.Vocabulary(D, rng=np.random.RandomState(seed=seed))
A = vocab.parse('A').v
B = vocab.parse('B').v
ideal_result = vocab.parse('A*B').v

data_A = []
data_B = []
data_C = []
for i in range(steps):
    output_values = {}

    # inject inputs from the outside world
    input_values[input_A][:] = A
    input_values[input_B][:] = B

    for n in model.all_nodes:
        output_values[n] = input_values[n].copy()
        input_values[n][:] = 0
    for ens in model.all_ensembles:
        output_values[ens] = cores[ens].step(input_values[ens])
        input_values[ens][:] = 0

    for m in messages:
        m.apply(input_values, output_values)

    #save some data for plotting
    data_A.append(output_values[model.inA.state.output].copy())
    data_B.append(output_values[model.inB.state.output].copy())
    data_C.append(output_values[model.result.state.output].copy())


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
filt_result = filt.filt(data_C, dt=sim.dt, y0=0)
accuracy = [np.dot(ideal_result/np.linalg.norm(ideal_result), 
                   x/np.linalg.norm(x)) for x in filt_result]
pylab.plot(accuracy)
pylab.ylim(-1,1)
pylab.show()







            

        


    

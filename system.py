import nengo
import numpy as np

import core

class Message(object):
    def __init__(self, 
                 pre_obj, pre_start, pre_len, 
                 post_obj, post_start, post_len,
                 matrix, synapse, dt):
        self.pre_obj = pre_obj        # object to read from
        self.pre_start = pre_start    # index to start reading from
        self.pre_len = pre_len        # number of values to read
        self.post_obj = post_obj      # object to send to
        self.post_start = post_start  # index to start sending to
        self.post_len = post_len      # number of values to send
        self.matrix = matrix          # matrix multiply to apply (or scalar)

        # optional low-pass filter to apply
        if synapse is None:
            self.synapse_scale = None
        else:
            self.synapse_scale = np.exp(-dt/synapse)
        self.value = np.zeros(post_len)

    def apply(self, input_values, output_values):
        # grab the full output
        v = output_values[self.pre_obj]
        # just grab the slice we need
        v = v[self.pre_start:self.pre_start+self.pre_len]

        # apply the matrix multiply (note this may just be a scalar or even 1)
        v = np.dot(self.matrix, v)

        # apply the low-pass filter
        if self.synapse_scale is not None:
            self.value = (1-self.synapse_scale)*self.value + self.synapse_scale*v
            v = self.value

        # set the result
        target = input_values[self.post_obj]
        target[self.post_start:self.post_start+self.post_len] += v


class Intermediate(object):
    def __init__(self, size):
        self.size = size

    def step(self, input):
        return input.copy()


class System(object):
    def __init__(self, model, sim):
        self.ens2core = {}          # mapping from nengo.Ensembles to Cores
        self.node2inter = {}        # mapping from nengo.Nodes to Intermediate

        self.input_values = {}      # the accumulated input for each Core
                                    #  and Intermediate object

        self.messages = []          # the set of messages that must be sent
                                    #  each time step

        self.extract_data_from_nengo(model, sim)

    def extract_data_from_nengo(self, model, sim):

        # find all the Connections out of an Ensemble
        conns_out = {}
        for ens in model.all_ensembles:
            conns_out[ens] = []
        for c in model.all_connections:
            if isinstance(c.pre_obj, nengo.Ensemble):
                conns_out[c.pre_obj].append(c)

        conn_dec_range = {}
        for ens in model.all_ensembles:
            funcs = []
            dec = []
            size = 0
            ranges = []
            # build up a decoder for all the computations from this Ensemble
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

            # generate the Core for this Ensemble
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
            self.ens2core[ens] = c
            self.input_values[c] = np.zeros(c.n_inputs)

        # generate Intermediate objects for each Node
        for n in model.all_nodes:
            assert n.output is None
            inter = Intermediate(n.size_in)
            self.node2inter[n] = inter
            self.input_values[inter] = np.zeros(inter.size)

        # generate all of the Messages
        for conn in model.all_connections:
            pre_obj = conn.pre_obj
            if conn.function is None:
                pre_indices = np.arange(pre_obj.size_out)[conn.pre_slice]
            else:
                pre_indices = np.arange(conn.size_mid)

            if isinstance(pre_obj, nengo.Ensemble):
                r = conn_dec_range[conn]
                indices = np.arange(self.ens2core[pre_obj].n_outputs)
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

            pre_obj = self.ens2core.get(pre_obj, pre_obj)
            post_obj = self.ens2core.get(post_obj, post_obj)
            pre_obj = self.node2inter.get(pre_obj, pre_obj)
            post_obj = self.node2inter.get(post_obj, post_obj)

            m = Message(pre_obj, pre_start, pre_len,
                        post_obj, post_start, post_len,
                        matrix=conn.transform,
                        synapse=(None if conn.synapse is None
                                 else conn.synapse.tau),
                        dt=sim.dt)
            self.messages.append(m)




    def step(self):
        output_values = {}

        # Do the core inner loop.  These 3 steps could be done in parallel!

        # process all the Intermediates
        for n in self.node2inter.values():
            output_values[n] = n.step(self.input_values[n])
            self.input_values[n][:] = 0

        # process all the Cores
        for c in self.ens2core.values():
            output_values[c] = c.step(self.input_values[c])
            self.input_values[c][:] = 0

        # send all the Messages
        for m in self.messages:
            m.apply(self.input_values, output_values)

        # return this so we can plot values from it
        return output_values

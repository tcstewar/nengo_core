import numpy as np

class Core(object):
    def __init__(self, n_inputs, n_neurons, n_outputs,
                 encoders, bias, decoders,
                 tau_rc=0.02, tau_ref=0.002,
                 dt=0.001, 
                 learning_rate=1e-4, learning_filter=0.01):
        assert n_inputs == encoders.shape[1]
        assert n_neurons == encoders.shape[0]
        assert n_neurons == bias.shape[0]
        assert n_neurons == decoders.shape[1]
        assert n_outputs == decoders.shape[0]

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs

        self.encoders = encoders         # the input weights
        self.bias = bias                 # constant bias input
        self.decoders = decoders / dt    # the output weights

        self.tau_rc = tau_rc             # membrane time constant
        self.tau_ref = tau_ref           # refractory period

        self.learning_rate = learning_rate / self.n_neurons / dt
        self.dt = dt

        self.voltage = np.zeros(n_neurons)         # membrane voltage
        self.refractory_time = np.zeros(n_neurons) # time in refractory period

        self.learning_activity = np.zeros(n_neurons)
        self.learning_scale = np.exp(-dt/learning_filter)

    def step(self, state, error=None):
        # feed input over the static synapses
        current = self.compute_neuron_input(state)
        # do the neural nonlinearity
        activity = self.neuron(current)
        # apply the learned synapses
        value = self.compute_output(activity)

        if error is not None:
            # apply low-pass filter on activity to smooth learning
            self.learning_activity = (activity*self.learning_scale +
                    self.learning_activity*(1-self.learning_scale))

            # update the synapses with the learning rule
            delta = np.outer(error, self.learning_activity)
            self.decoders -= delta * self.learning_rate

        return value

    def neuron(self, current):
        # reduce all refractory times by dt
        self.refractory_time -= self.dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep
        delta_t = (self.dt - self.refractory_time).clip(0, self.dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        self.voltage -= (current - self.voltage) * np.expm1(-delta_t / self.tau_rc)

        # determine which neurons spiked this time step
        #  NOTE: this will be very sparse, since few neurons spike at once
        spiked = self.voltage > 1

        # compute when during the timestep the spike happened
        t_spike = self.dt + self.tau_rc * np.log1p(
                    -(self.voltage[spiked] - 1) / (current[spiked] - 1))
        # use this time to set the refractory_time accurately
        self.refractory_time[spiked] = self.tau_ref + t_spike

        # set spiked voltages to zero, and rectify negative voltages to zero
        self.voltage[self.voltage < 0] = 0
        self.voltage[spiked] = 0

        return spiked

    def compute_neuron_input(self, state):
        # this tends to be a dense matrix multiply
        return np.dot(self.encoders, state) + self.bias

    def compute_output(self, activity):
        # since activity is sparse (1s for neurons who just spiked, 0s for
        #  others), this may be efficiently done as an add operation
        return np.dot(self.decoders, activity)


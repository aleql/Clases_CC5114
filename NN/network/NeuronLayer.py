import numpy as np

from neurons.PerceptronNeuron import PerceptronNeuron
from neurons.SigmoidNeuron import SigmoidNeuron


class NeuronLayer:

    def __init__(self, type, n_neurons, input_size):
        self.neurons = []
        # Generate neurons
        for i in range(n_neurons):
            # Generate n_neurons
            Ws = np.random.uniform(-2, 2, input_size)
            bias = np.random.uniform(-2, 2)
            if type == 'relu':
                neuron = PerceptronNeuron(Ws, bias)
            elif type == 'sigmoid':
                neuron = SigmoidNeuron(Ws, bias)
            self.neurons.append(neuron)

    def feed(self, input):
        output = []
        for neuron in self.neurons:
            output.append(neuron.feed(input))
        return output

    def get_layer_output(self):
        output = []
        for neuron in self.neurons:
            output.append(neuron.output)
        return output

    def adjustDeltaLayer(self, error_list):
        for neuron, error in zip(self.neurons, error_list):
            neuron.adjustDelta(error)

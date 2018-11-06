import numpy as np

from neurons.PerceptronNeuron import PerceptronNeuron
from neurons.SigmoidNeuron import SigmoidNeuron


class NeuronLayer:

    def __init__(self, type, n_neurons, input_size):
        self.neurons = []
        self.layer_output = []
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
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.feed(input))
        self.layer_output.append(outputs)
        return outputs

    def adjustDeltaLayer(self, error):
        for neuron in self.neurons:
            neuron.adjustDelta(error)


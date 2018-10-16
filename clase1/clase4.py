import numpy as np

from clase3 import PerceptronNeuron, SigmoidNeuron


class NeuronLayer:

    def __init__(self, type, neurons, input_size):
        self.neurons = []
        # Generate neurons
        for i in range(neurons):
            # Generate parameters
            Ws = np.random.uniform(-2, 2, input_size)
            bias = np.random.uniform(-2, 2)
            if type == 'relu':
                neuron = PerceptronNeuron(Ws, bias)
            elif type == 'sigmoid':
                neuron = SigmoidNeuron(Ws, bias)
            neurons.append(neuron)

    def feed(self, input):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.feed(input))
        return outputs


class NeuralNetworks:

    def __init(self):
        self.layers = None

    def forward(self, input):
        output = None
        for layer in self.layers:
            if output is None:
                output = layer.feed(input)
            else:
                output = layer.feed(input)
        return output



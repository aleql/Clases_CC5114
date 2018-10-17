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

    def train(self, input, expectedOutput):
        self.forward(input)
        self.backawardPropagationError(expectedOutput)


    def backawardPropagationError(self, expectedOutput):
        for layer_index in range(len(self.layers) - 1, -1, -1):
            if layer_index == len(self.layers) - 1: # Ultima capa
                error = expectedOutput - self.layers[layer_index].output
                transferDerivate = np.dot(self.layers[layer_index].output, (1.0 - self.layers[layer_index].output))
                self.layers[layer_index].neurons[0].delta = error * transferDerivate
            else: # No es la ultima capa
                error = 0.0
                for neuron in self.layers[layer_index]:
                    for next_neuron in self.layers[layer_index + 1].neurons: # Neuronas de la capa siguiente
                        error += next_neuron.weights[j] * next_neuron.delta






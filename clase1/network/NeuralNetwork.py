import numpy as np

from auxiliar_methods.statistics_errors import simple_error
from network.NeuronLayer import NeuronLayer


class NeuralNetwork:

    def __init__(self, layers_structure, neuron_type, input_size):
        self.layers = []
        self.learning_rate = 0.5
        neurons_in_previous_layer = None
        for neurons_in_layer in layers_structure:
            # Inputs for the layer is the number of outputs in previous layer
            input_size = input_size if neurons_in_previous_layer is None else neurons_in_previous_layer

            self.layers.append(NeuronLayer(neuron_type, neurons_in_layer, input_size))

            neurons_in_previous_layer = neurons_in_layer

    def train(self, input, expectedOutput, nEpochs=1):
        for i in range(nEpochs):
            self.forward(input)
            self.backawardPropagationError(expectedOutput)
            self.updateWeights(input)

    def forward(self, input):
        output = None
        for layer in self.layers:
            if output is None:
                output = layer.feed(input)
            else:
                output = layer.feed(output)
        return output

    def backawardPropagationError(self, expectedOutput):
        print(len(self.layers) - 1)

        for layer_index in range(len(self.layers) - 1, -1, -1):

            # Last layer
            if layer_index == len(self.layers) - 1:
                error = simple_error(expectedOutput, self.layers[layer_index].layer_output)
                self.layers[layer_index].adjustDeltaLayer(error)

            # Hidden layer
            else:
                connection_index = 0  #
                for neuron in self.layers[layer_index].neurons:
                    error = 0.0
                    for next_neuron in self.layers[layer_index + 1].neurons:  # Neuronas de la capa siguiente
                        error += next_neuron.weights[connection_index] * next_neuron.delta
                    # Update delta
                    neuron.adjustDelta(error)
                    connection_index += 1

    def updateWeights(self, input):
        first = True
        for layer_index in range(len(self.layers) - 1):
            # Use input or output of previous layer
            if first:
                input = input
                first = False
            else:
                previous_layer = self.layers[layer_index - 1]

                # Collect outputs:
                input = previous_layer.layer_output

            for neuron in self.layers[layer_index].neurons:
                neuron.adjustWeight(input, self.learning_rate)
                neuron.adjustBias(self.learning_rate)

                # for neuron_index in range(len(self.layers[layer_index].neurons)):
                #     self.layers[layer_index].neurons[neuron_index].adjustWeight(input, self.learning_rate)
                #     self.layers[layer_index].neurons[neuron_index].adjustBias(input, self.learning_rate)
                #
                #

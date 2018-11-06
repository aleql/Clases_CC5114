import numpy as np

from auxiliar_methods.statistics_errors import simple_error
from network.NeuronLayer import NeuronLayer


class NeuralNetwork:

    def __init__(self, layers_structure, neuron_type, input_size):
        self.layers = []
        self.learning_rate = 0.5
        self.NN_outputs = [[], []]
        neurons_in_previous_layer = None
        for neurons_in_layer in layers_structure:
            # Inputs for the layer is the number of outputs in previous layer
            input_size = input_size if neurons_in_previous_layer is None else neurons_in_previous_layer
            self.layers.append(NeuronLayer(neuron_type, neurons_in_layer, input_size))
            neurons_in_previous_layer = neurons_in_layer

    def train(self, inputs, expectedOutputs):

        outputs = []
        for input in inputs:
            outputs.append(self.forward(input))

        self.backawardPropagation(expectedOutputs)
        for input in inputs:
            self.updateWeights(input)

    def forward(self, input):
        output = None
        for layer in self.layers:
            if output is None:
                output = layer.feed(input)
            else:
                output = layer.feed(output)
        return output

    def backawardPropagation(self, expectedOutputs):
        for expectedOutput in expectedOutputs:
            self.backawardPropagationError(expectedOutput)

    def backawardPropagationError(self, expectedOutput):

        for layer_index in range(len(self.layers) - 1, -1, -1):

            # Last layer
            if layer_index == len(self.layers) - 1:

                layer_output = self.layers[layer_index].layer_output.pop(0)

                # Append outputs for getting error of train
                self.NN_outputs[0] += expectedOutput
                self.NN_outputs[1] += layer_output

                # Calculate error and adjust delta
                error = simple_error(expectedOutput, layer_output)
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
        for layer_index in range(len(self.layers)):
            # Use input or output of previous layer
            if first:
                input = input
                first = False
            else:
                previous_layer = self.layers[layer_index - 1]

                # Collect outputs:
                input = previous_layer.layer_output.pop()

            for neuron in self.layers[layer_index].neurons:
                neuron.adjustWeight(input, self.learning_rate)
                neuron.adjustBias(self.learning_rate)

    def get_error(self):
        # Calculate error
        error = simple_error(self.NN_outputs[0], self.NN_outputs[1])
        # Set to default
        self.NN_outputs = [[], []]
        return error


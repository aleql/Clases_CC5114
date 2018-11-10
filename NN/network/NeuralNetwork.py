import numpy as np

from auxiliar_methods.statistics_errors import simple_error, cuadratic_error, accuracy_net, identity
from network.NeuronLayer import NeuronLayer


class NeuralNetwork:

    def __init__(self, layers_structure, neuron_type, input_size, learning_rate=0.5):
        self.layers = []
        self.learning_rate = learning_rate
        neurons_in_previous_layer = None
        for neurons_in_layer in layers_structure:
            # Inputs for the layer is the number of outputs in previous layer
            input_size = input_size if neurons_in_previous_layer is None else neurons_in_previous_layer
            self.layers.append(NeuronLayer(neuron_type, neurons_in_layer, input_size))
            neurons_in_previous_layer = neurons_in_layer

    def train(self, inputs, expectedOutputs):

        outputs = []
        for input, expectedOutput in zip(inputs, expectedOutputs):
            outputs.append(self.forward(input))
            self.backawardPropagationError(expectedOutput)
            self.updateWeights(input)

        return self.get_stats(outputs, expectedOutputs)

    def eval(self, inputs, expectedOutputs):
        outputs = []
        for input in inputs:
            outputs.append(self.forward(input))
        return self.get_stats(outputs, expectedOutputs)


    def forward(self, input):
        output = None
        for layer in self.layers:
            if output is None:
                output = layer.feed(input)
            else:
                output = layer.feed(output)
        return output

    def backawardPropagationError(self, expectedOutput):

        for layer_index in range(len(self.layers) - 1, -1, -1):

            # Last layer
            if layer_index == len(self.layers) - 1:

                # Get layer output
                layer_output = self.layers[layer_index].get_layer_output()

                # error list:
                error_list = []
                for exp_output, real_output in zip(expectedOutput, layer_output):
                    error_list.append(float(exp_output - real_output))
                self.layers[layer_index].adjustDeltaLayer(error_list)

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
            # print("layer {}".format(layer_index))
            # Use input or output of previous layer
            if first:
                input = input
                first = False
            else:
                previous_layer = self.layers[layer_index - 1]

                # Collect outputs:
                input = previous_layer.get_layer_output()

            neuron_index = 0
            for neuron in self.layers[layer_index].neurons:
                # print("  neuron {}".format(neuron_index))
                neuron.adjustWeight(input, self.learning_rate)
                neuron.adjustBias(self.learning_rate)
                neuron_index += 1

    def get_stats(self, outputs, expected_outputs, error_type='simple'):
        # Calculate error
        if error_type == 'simple':
            error = simple_error(expected_outputs, outputs)
        elif error_type == 'cuadratic':
            error = cuadratic_error(expected_outputs, outputs)
        # Get accuracy
        accuracy = accuracy_net(expected_outputs, outputs)
        return error, accuracy

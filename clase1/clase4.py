import numpy as np

from clase3 import PerceptronNeuron, SigmoidNeuron


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
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.feed(input))
        return outputs


class NeuralNetworks:

    def __init__(self, layers, n_neurons, neuron_type, input_size):
        self.layers = []
        self.learning_rate = 0.5
        for i in range(0, layers):
            self.layers.append(NeuronLayer(neuron_type, n_neurons, input_size))


    def forward(self, input):
        output = None
        for layer in self.layers:
            if output is None:
                output = layer.feed(input)
            else:
                output = layer.feed(input)
        return output

    def train(self, input, expectedOutput, nEpochs):
        self.forward(input)
        self.backawardPropagationError(expectedOutput)
        self.updateWeights(input)

    def backawardPropagationError(self, expectedOutput):
        for layer_index in range(len(self.layers) - 1, -1, -1):
            if layer_index == len(self.layers) - 1:  # Ultima capa
                error = expectedOutput - self.layers[layer_index].output
                transferDerivate = np.dot(self.layers[layer_index].output, (1.0 - self.layers[layer_index].output))
                self.layers[layer_index].neurons[0].delta = error * transferDerivate
            else:  # No es la ultima capa
                error = 0.0
                for neuron in self.layers[layer_index]:
                    j = 0
                    for next_neuron in self.layers[layer_index + 1].neurons:  # Neuronas de la capa siguiente
                        error += next_neuron.weights[j] * next_neuron.delta
                        j += 1
                # Update delta
                transfer_derivate = neuron.output * (1.0 - neuron.output)
                neuron.delta = neuron.delta * error

    def updateWeights(self, input):
        first = True
        for layer_index in range(len(self.layers) - 1, 1):
            layer = self.layers[layer_index]
            if first:
                input = input
                first = False
            else:
                previous_layer = self.layers[layer_index - 1]

                # Collect outputs:
                output_list = []
                for prev_neuron in previous_layer:
                    output_list.append(prev_neuron.output)
                previous_output = np.vstack(output_list)

            for neuron in layer:
                neuron.adjustWeight(input, self.learning_rate)
                neuron.adjustBias(input, self.learning_rate)



# precision: lo hizo bien vs todo
# recall, cuantas buenas respuestas ha tenido del total la red
# a mas epochs menos error
# mismo rango en en input que la fncion de activacion
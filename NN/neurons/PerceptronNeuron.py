from neurons.AbstractNeuron import AbstractNeuron

# Define relu activation function
relu = lambda x: 1 if x > 0 else 0


class PerceptronNeuron(AbstractNeuron):

    def __init__(self, weights, bias):
        super().__init__(weights, bias)
        self.weights = weights
        self.bias = bias
        self.activation_function = relu

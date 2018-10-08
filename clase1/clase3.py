import numpy as np


class AbstractNeuron:

    def __init__(self, weights, bias):
        self.activation_function = None
        self.weights = weights
        self.bias = bias
        super().__init__()


    def feed(self, inputs):
        if len(inputs.shape) == 1:
            computation = np.dot(inputs, self.weights) + self.bias
            return (self.activation_function(computation))
        else:
            computationArray = []
            for i in range(0, len(inputs)):
                computation = np.dot(inputs[i], self.weights) + self.bias
                computationArray.append(computation)
            computationArray = list(map(self.activation_function, computationArray))
            return np.array(computationArray)

    def train(self, desiredOutput, realOutput, lr, inputs):
        diff = desiredOutput - realOutput
        for i in range(0, len(inputs)):
            self.weights[i] = self.weights[i] + (lr * np.dot(inputs[i], diff))
        self.bias = self.bias + np.dot(np.full(len(diff), lr), diff)




class Sigmoid(AbstractNeuron):

    def __init__(self, weights, bias):
        super().__init__(weights, bias)
        self.weights = weights
        self.bias = bias
        self.activation_function = lambda x : 1 / (1 + np.math.exp(-x))








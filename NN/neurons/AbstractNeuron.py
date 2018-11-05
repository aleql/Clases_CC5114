from functools import reduce

import numpy as np


class AbstractNeuron:

    def __init__(self, weights, bias):
        self.activation_function = None
        self.weights = weights
        self.bias = bias
        self.output = None
        self.delta = None # Partial Derivate
        super().__init__()


    def feed(self, inputs):
        computation_output = 0
        for input, weight in zip(inputs, self.weights):
            computation_output += input * weight
        computation_output = (self.activation_function(computation_output + self.bias))
        self.output = computation_output
        return computation_output

    def train(self, desiredOutput, realOutput, lr, inputs):
        # diff of both vectors
        # diff = [x - y for x, y in zip(desiredOutput, realOutput)]
        diff = desiredOutput - realOutput
        # Update parameters
        for i in range(0, len(inputs)):
            self.weights[i] = self.weights[i] + (lr * inputs[i] * diff)
        self.bias = self.bias + lr * diff

    def adjustDelta(self, error):
        self.delta = error * self.transferDerivate()

    def adjustBias(self, learning_rate):
        self.bias += (learning_rate * self.delta)

    def adjustWeight(self, input, learning_rate):
        for i in range(0, len(input)):
            self.weights[i] = self.weights[i] + (learning_rate * np.dot(input[i], self.delta))

    def transferDerivate(self):
        return self.output * (1.0 - self.output)





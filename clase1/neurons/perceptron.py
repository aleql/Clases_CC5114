from functools import reduce

import numpy as np


class perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed(self, inputs):
        computation_output = 0
        for input, weight in zip(inputs, self.weights):
            computation_output += input * weight
        computation_output += self.bias
        return 1 if computation_output > 0 else 0

    def train(self, desiredOutput, realOutput, lr, inputs):
        # diff of both vectors
        diff = [x - y for x, y in zip(desiredOutput, realOutput)]
        # Update parameters
        for i in range(0, len(inputs)):
            self.weights[i] = self.weights[i] + (lr * inputs[i] * diff)
        self.bias = self.bias + lr * [reduce((lambda x, y: x + y), diff)]



# main
perceptron_and = perceptron([1, 1], -1.5)
perceptron_or = perceptron([1, 1], -0.5)
perceptron_nand = perceptron([-2, -2], 3)

# sum with carry
def p_sum(x1, x2):
    perceptron_nand = perceptron([-2, -2], 3)
    out_1 = perceptron_nand.feed([x1, x2])
    out_2 = perceptron_nand.feed([x1, out_1])
    out_3 = perceptron_nand.feed([x2, out_1])
    out_carry = perceptron_nand.feed([out_1, out_1])
    out_sum = perceptron_nand.feed([out_2, out_3])
    return out_sum, out_carry












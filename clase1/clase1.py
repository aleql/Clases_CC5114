import numpy as np


class perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed(self, inputs):
        if len(inputs.shape) == 1:
            computation = np.dot(inputs, self.weights) + self.bias
            if computation <= 0:
                return 0
            else:
                return 1
        else:
            computationArray = []
            for i in range(0, len(inputs)):
                computation = np.dot(inputs[i], self.weights) + self.bias
                computationArray.append(computation)
            computationArray = list(map(lambda x: 0 if x < 0 else 1, computationArray))
            return np.array(computationArray)

    def train(self, desiredOutput, realOutput, lr, inputs):
        diff = desiredOutput - realOutput
        for i in range(0, len(inputs)):
            self.weights[i] = self.weights[i] + (lr * np.dot(inputs[i], diff))
        self.bias = self.bias + np.dot(np.full(len(diff), lr), diff)



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












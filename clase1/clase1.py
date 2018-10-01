import numpy

class perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed(self, inputs):
        computation = numpy.dot(inputs, self.weights) + self.bias
        if computation <= 0:
            return 0
        else:
            return 1


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











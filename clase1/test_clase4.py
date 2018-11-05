import numpy as np

from network.NeuralNetwork import NeuralNetwork

# sigmoid_NN = NeuralNetwork([3, 3, 1], 'sigmoid', 10)
# test_input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# output = sigmoid_NN.forward(test_input)
#
#
# print(output)

# Test case 1
# 1 hidden layer inputs size 2 ; of 2 neurons and 1 ouput
sigmoid_NN = NeuralNetwork([1, 1], 'sigmoid', 2)

# Change bias/weights manually

# First hidden layer, one neuron, two inputs one ouput
sigmoid_NN.layers[0].neurons[0].bias = 0.5
sigmoid_NN.layers[0].neurons[0].weights = [0.4, 0.3]

# Output layer, one input one output
sigmoid_NN.layers[1].neurons[0].bias = 0.4
sigmoid_NN.layers[1].neurons[0].weights = [0.3]

sigmoid_NN.train([1, 1], [1])

print("neuron 1 bias: {}".format(sigmoid_NN.layers[0].neurons[0].bias))
print("neuron 1 weights: {}".format(sigmoid_NN.layers[0].neurons[0].weights))

print("neuron 2 bias: {}".format(sigmoid_NN.layers[1].neurons[0].bias))
print("neuron 2 weights: {}".format(sigmoid_NN.layers[1].neurons[0].weights))

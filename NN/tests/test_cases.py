import numpy as np

from network.NeuralNetwork import NeuralNetwork

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

print("--- Test case | ---")

print("neuron 1 bias: {}".format(sigmoid_NN.layers[0].neurons[0].bias))
print("neuron 1 weights: {}".format(sigmoid_NN.layers[0].neurons[0].weights))

print("neuron 2 bias: {}".format(sigmoid_NN.layers[1].neurons[0].bias))
print("neuron 2 weights: {}".format(sigmoid_NN.layers[1].neurons[0].weights))


# Test case 2
# 1 hidden layer of size 2, and one output layer of size 2
sigmoid_NN = NeuralNetwork([2, 2], 'sigmoid', 2)

# Change bias/weights manually

# First hidden layer, two neurons, two inputs two ouputs
sigmoid_NN.layers[0].neurons[0].bias = 0.5
sigmoid_NN.layers[0].neurons[0].weights = [0.7, 0.3]
sigmoid_NN.layers[0].neurons[1].bias = 0.4
sigmoid_NN.layers[0].neurons[1].weights = [0.3, 0.7]

# Output layer, two input and two outputs
sigmoid_NN.layers[1].neurons[0].bias = 0.3
sigmoid_NN.layers[1].neurons[0].weights = [0.2, 0.3]
sigmoid_NN.layers[1].neurons[1].bias = 0.6
sigmoid_NN.layers[1].neurons[1].weights = [0.4, 0.2]


sigmoid_NN.train([1, 1], [1, 1])

print("--- Test case || ---")

print("neuron 1 bias: {}".format(sigmoid_NN.layers[0].neurons[0].bias))
print("neuron 1 weights: {}".format(sigmoid_NN.layers[0].neurons[0].weights))
print("neuron 2 bias: {}".format(sigmoid_NN.layers[0].neurons[1].bias))
print("neuron 2 weights: {}".format(sigmoid_NN.layers[0].neurons[1].weights))

print("neuron 3 bias: {}".format(sigmoid_NN.layers[1].neurons[0].bias))
print("neuron 3 weights: {}".format(sigmoid_NN.layers[1].neurons[0].weights))
print("neuron 4 bias: {}".format(sigmoid_NN.layers[1].neurons[1].bias))
print("neuron 4 weights: {}".format(sigmoid_NN.layers[1].neurons[1].weights))



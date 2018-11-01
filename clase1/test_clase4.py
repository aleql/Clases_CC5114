
from clase4 import NeuralNetworks
import numpy as np

sigmoid_NN = NeuralNetworks(3, 3, 'sigmoid', 10)
test_input = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
sigmoid_NN.forward(test_input)

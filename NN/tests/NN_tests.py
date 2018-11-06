import unittest

from auxiliar_methods.statistics_errors import truncate
from network.NeuralNetwork import NeuralNetwork


# Test case 1:
# 1 hidden layer inputs size 2 ; of 2 neurons and 1 ouput
class Test_case_1(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.sigmoid_NN = NeuralNetwork([1, 1], 'sigmoid', 2)

        # Change bias/weights manually

        # First hidden layer, one neuron, two inputs one ouput
        self.sigmoid_NN.layers[0].neurons[0].bias = 0.5
        self.sigmoid_NN.layers[0].neurons[0].weights = [0.4, 0.3]

        # Output layer, one input one output
        self.sigmoid_NN.layers[1].neurons[0].bias = 0.4
        self.sigmoid_NN.layers[1].neurons[0].weights = [0.3]

        # Train the NN
        self.sigmoid_NN.train([[1, 1]], [[1]])

    def test_parameters_train(self):
        assert truncate(self.sigmoid_NN.layers[0].neurons[0].bias, 3) == 0.502, "Test case 1: neuron 1 bias not updated " \
                                                                             "correctly. "
        assert truncate(self.sigmoid_NN.layers[0].neurons[0].weights[0], 3) == 0.402, "Test case 1: neuron 1 weight[0] " \
                                                                                   "not updated correctly. "
        assert truncate(self.sigmoid_NN.layers[0].neurons[0].weights[1], 3) == 0.302, "Test case 1: neuron 1 weight[1] " \
                                                                                   "not updated correctly. "

        assert truncate(self.sigmoid_NN.layers[1].neurons[0].bias, 3) == 0.439, "Test case 1: neuron 2 bias not updated " \
                                                                             "correctly. "
        assert truncate(self.sigmoid_NN.layers[1].neurons[0].weights[0], 3) == 0.330, "Test case 1: neuron 2 weight[0] " \
                                                                                   "not updated correctly. "


# Test case 2
# 1 hidden layer of size 2, and one output layer of size 2
class Test_case_2(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.sigmoid_NN = NeuralNetwork([2, 2], 'sigmoid', 2)

        # Change bias/weights manually

        # First hidden layer, two neurons, two inputs two ouputs
        self.sigmoid_NN.layers[0].neurons[0].bias = 0.5
        self.sigmoid_NN.layers[0].neurons[0].weights = [0.7, 0.3]
        self.sigmoid_NN.layers[0].neurons[1].bias = 0.4
        self.sigmoid_NN.layers[0].neurons[1].weights = [0.3, 0.7]

        # Output layer, two input and two outputs
        self.sigmoid_NN.layers[1].neurons[0].bias = 0.3
        self.sigmoid_NN.layers[1].neurons[0].weights = [0.2, 0.3]
        self.sigmoid_NN.layers[1].neurons[1].bias = 0.6
        self.sigmoid_NN.layers[1].neurons[1].weights = [0.4, 0.2]

        # Train the NN
        self.sigmoid_NN.train([[1, 1]], [[1, 1]])

    def test_parameters_train(self):

        assert truncate(self.sigmoid_NN.layers[0].neurons[0].bias, 4) == 0.5025, "Test case 2: neuron 1 bias not updated " \
                                                                            "correctly. "
        assert truncate(self.sigmoid_NN.layers[0].neurons[0].weights[0], 4) == 0.7025, "Test case 2: neuron 1 weight[0] " \
                                                                                  "not updated correctly. "
        assert truncate(self.sigmoid_NN.layers[0].neurons[0].weights[1], 4) == 0.3025, "Test case 2: neuron 1 weight[1] " \
                                                                                  "not updated correctly. "
        assert truncate(self.sigmoid_NN.layers[0].neurons[1].bias, 4) == 0.4024, "Test case 2: neuron 2 bias not updated " \
                                                                            "correctly. "
        assert truncate(self.sigmoid_NN.layers[0].neurons[1].weights[0], 4) == 0.3024, "Test case 2: neuron 2 weight[0] " \
                                                                                  "not updated correctly. "
        assert truncate(self.sigmoid_NN.layers[0].neurons[1].weights[1], 4) == 0.7024, "Test case 2: neuron 2 weight[1] " \
                                                                                  "not updated correctly. "
        assert truncate(self.sigmoid_NN.layers[1].neurons[0].bias, 4) == 0.3366, "Test case 2: neuron 3 bias not updated " \
                                                                            "correctly. "
        assert truncate(self.sigmoid_NN.layers[1].neurons[0].weights[0], 4) == 0.2299, "Test case 2: neuron 3 weight[0] " \
                                                                                  "not updated correctly. "
        assert truncate(self.sigmoid_NN.layers[1].neurons[0].weights[1], 4) == 0.3293, "Test case 2: neuron 3 weight[1] " \
                                                                                  "not updated correctly."
        assert truncate(self.sigmoid_NN.layers[1].neurons[1].bias, 4) == 0.6237, "Test case 2: neuron 4 bias not updated " \
                                                                            "correctly. "
        assert truncate(self.sigmoid_NN.layers[1].neurons[1].weights[0], 4) == 0.4194, "Test case 2: neuron 4 weight[0] " \
                                                                                  "not updated correctly. "
        assert truncate(self.sigmoid_NN.layers[1].neurons[1].weights[1], 4) == 0.2190, "Test case 2: neuron 4 weight[1] " \
                                                                                  "not updated correctly. "

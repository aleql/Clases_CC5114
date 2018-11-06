from network.NeuralNetwork import NeuralNetwork
from network.NN_controller import NN_controller

data = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]

exp_outputs = [[0], [0], [0], [1]]

sigmoid_NN = NeuralNetwork([3, 1], 'sigmoid', 3)

controller = NN_controller(sigmoid_NN)

controller.train_epochs(data, exp_outputs, 1)
print(controller.NeuralNetwork.get_error())

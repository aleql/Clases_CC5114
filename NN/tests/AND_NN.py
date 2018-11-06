from network.NeuralNetwork import NeuralNetwork
from network.NN_controller import NN_controller

data = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]

exp_outputs = [[0], [0], [0], [1]]

sigmoid_NN = NeuralNetwork([1, 1], 'relu', 3)

controller = NN_controller(sigmoid_NN, 'cuadratic')

results = controller.train_epochs(data, exp_outputs, 2)
print(results[-1])

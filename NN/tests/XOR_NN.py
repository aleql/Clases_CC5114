from network.NeuralNetwork import NeuralNetwork
from network.NN_controller import NN_controller
import matplotlib.pyplot as plt

data = [[0, 0], [0, 1], [1, 0], [1, 1]]

exp_outputs = [[0], [1], [1], [0]]

sigmoid_NN = NeuralNetwork([2, 1], 'relu', 2)

controller = NN_controller(sigmoid_NN, 'simple')

results = controller.train_epochs(data, exp_outputs, 100)
plt.plot(results)
plt.show()
# print(results[-1])

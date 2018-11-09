import os

from Seed_NN.CSV_loader import CSV_loader
from auxiliar_methods.statistics_errors import index_to_list
from network.NN_controller import NN_controller
from network.NeuralNetwork import NeuralNetwork
from matplotlib import pyplot as plt

# Load csv
dataset_dir = os.path.join(os.path.abspath(os.path.join(__file__ , "../../..")), os.path.join('datasets', 'seeds_dataset.csv'))
csv_df = CSV_loader(dataset_dir, 7, index_to_list)
input_length = len(csv_df.train.inputs[0])


# # Create the NN
# # 1 inputs layer, 2 hidden layers, 1 output
# seed_NN = NeuralNetwork([input_length, input_length - 1, input_length - 2, 3], 'sigmoid', input_length)
#
# controller = NN_controller(seed_NN, 'simple')
#
# results = controller.train_epochs(csv_df.inputs, csv_df.expected_outputs, 10)
# plt.plot(results)
# plt.show()
#

# Create every NN combination and check results

# 1 hidden layer
# learning rate 0.5
# number of neurons from input size to 1
errors_results = []
accuracies_results = []
for i in range(input_length):
    print("Number of neurns {}".format(i))
    seed_NN = NeuralNetwork([i, 3], 'sigmoid', input_length)
    controller = NN_controller(seed_NN, 'simple')

    # Train
    controller.train_epochs(csv_df.train.inputs, csv_df.train.expected_outputs, 100)

    # Tests
    errors, accuracies = controller.test(csv_df.test.inputs, csv_df.test.expected_outputs)

    errors_results.append(errors)
    accuracies_results.append(accuracies)

for accuracy_result in accuracies_results:
    plt.plot(accuracy_result)
    plt.show()


# 2 hidden layers


# 3 hidden layers


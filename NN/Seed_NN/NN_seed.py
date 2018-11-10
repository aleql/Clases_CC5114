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


# Create every NN combination and check results
learning_rate = 0.5

# Number of epochs
epochs = 100
# 1 hidden layer
# learning rate 0.5
# number of neurons from input size to 1
errors_results = []
accuracies_results = []
number_of_epochs = list(range(10, epochs + 10, 10))
seed_NN = NeuralNetwork([7, 3], 'sigmoid', input_length, learning_rate=learning_rate)
controller = NN_controller(seed_NN, 'cuadratic')

for epoch in number_of_epochs:
    # Train
    controller.train_epochs(csv_df.train.inputs, csv_df.train.expected_outputs, 10)

    # Tests
    errors, accuracies = controller.eval(csv_df.test.inputs, csv_df.test.expected_outputs)

    errors_results.append(errors)
    accuracies_results.append(accuracies)

    # # Shuffle dataset, loading it again
    # dataset_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")),
    #                            os.path.join('datasets', 'seeds_dataset.csv'))
    # csv_df = CSV_loader(dataset_dir, 7, index_to_list)
    # input_length = len(csv_df.train.inputs[0])


# error chart
plt.plot(number_of_epochs, errors_results)
plt.xlabel('number of epochs')
plt.ylabel('mean cuadratic error')
plt.title('Error by number of epochs for NN 1 hidden layer and lr {}'.format(learning_rate))
plt.show()

# accuracy chart
plt.plot(number_of_epochs, accuracies_results)
plt.xlabel('number of epochs')
plt.ylabel('accuracy')
plt.title('Accuracy by number of epochs for NN 1 hidden layer and lr {}'.format(learning_rate))
plt.show()


errors_results = []
accuracies_results = []



# 2 hidden layers

# 1 hidden layer
# learning rate 0.5
results_dir = os.path.join(os.path.abspath(os.path.join(__file__ , "../..")), os.path.join('results', '05'))
errors_results = []
accuracies_results = []
number_of_epochs = list(range(10, epochs + 10, 10))


seed_NN = NeuralNetwork([7, 6, 3], 'sigmoid', input_length)
controller = NN_controller(seed_NN, 'cuadratic')

for epoch in number_of_epochs:
    # Train
    controller.train_epochs(csv_df.train.inputs, csv_df.train.expected_outputs, 10)

    # Tests
    errors, accuracies = controller.eval(csv_df.test.inputs, csv_df.test.expected_outputs)

    errors_results.append(errors)
    accuracies_results.append(accuracies)

    # # Shuffle dataset, loading it again
    # dataset_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")),
    #                            os.path.join('datasets', 'seeds_dataset.csv'))
    # csv_df = CSV_loader(dataset_dir, 7, index_to_list)
    # input_length = len(csv_df.train.inputs[0])
    #


# error chart
plt.plot(number_of_epochs, errors_results)
plt.xlabel('number of epochs')
plt.ylabel('mean cuadratic error')
plt.title('Error by number of epochs for NN 2 hidden layers and lr {} with shuffle'.format(learning_rate))
plt.show()

# accuracy chart
plt.plot(number_of_epochs, accuracies_results)
plt.xlabel('number of epochs')
plt.ylabel('accuracy')
plt.title('Accuracy by number of epochs for NN 2 hidden layers and lr {}'.format(learning_rate))
plt.show()


errors_results = []
accuracies_results = []



# 3 hidden layers
results_dir = os.path.join(os.path.abspath(os.path.join(__file__ , "../..")), os.path.join('results', '05'))
errors_results = []
accuracies_results = []
number_of_epochs = list(range(10, epochs + 10, 10))


seed_NN = NeuralNetwork([7, 6, 5, 3], 'sigmoid', input_length)
controller = NN_controller(seed_NN, 'cuadratic')

for epoch in number_of_epochs:
    # Train
    controller.train_epochs(csv_df.train.inputs, csv_df.train.expected_outputs, 10)

    # Tests
    errors, accuracies = controller.eval(csv_df.test.inputs, csv_df.test.expected_outputs)

    errors_results.append(errors)
    accuracies_results.append(accuracies)

    # # Shuffle dataset, loading it again
    # dataset_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")),
    #                            os.path.join('datasets', 'seeds_dataset.csv'))
    # csv_df = CSV_loader(dataset_dir, 7, index_to_list)
    # input_length = len(csv_df.train.inputs[0])



# error chart
plt.plot(number_of_epochs, errors_results)
plt.xlabel('number of epochs')
plt.ylabel('mean cuadratic error')
plt.title('Error by number of epochs for NN 3 hidden layers and lr {}'.format(learning_rate))
plt.show()

# accuracy chart
plt.plot(number_of_epochs, accuracies_results)
plt.xlabel('number of epochs')
plt.ylabel('accuracy')
plt.title('Accuracy by number of epochs for NN 3 hidden layers and lr {}'.format(learning_rate))
plt.show()


errors_results = []
accuracies_results = []


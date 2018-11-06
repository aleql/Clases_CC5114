import os

from Seed_NN.CSV_loader import CSV_loader
from network.NN_controller import NN_controller
from network.NeuralNetwork import NeuralNetwork

# Load csv
dataset_dir = os.path.join(os.path.abspath(os.path.join(__file__ , "../../..")), os.path.join('datasets', 'seeds_dataset.csv'))
csv_df = CSV_loader(dataset_dir, 7)
input_length = len(csv_df.inputs[0])


# Create the NN
# 1 inputs layer, 2 hidden layers, 1 output
seed_NN = NeuralNetwork([input_length, input_length - 1, input_length - 2, 1], 'sigmoid', input_length)

controller = NN_controller(seed_NN, 'cuadratic')

results = controller.train_epochs(csv_df.inputs, csv_df.expected_outputs, 100)

print(results[-1])
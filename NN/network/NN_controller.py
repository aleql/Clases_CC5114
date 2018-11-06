
class NN_controller:

    def __init__(self, NeuralNetwork):
        self.NeuralNetwork = NeuralNetwork
        self.error_per_epoch = []

    def train_epochs(self, input_set, exp_output_set, epochs):
        for i in range(epochs):
            self.NeuralNetwork.train(input_set, exp_output_set)
            self.error_per_epoch.append(self.NeuralNetwork.get_error())
        return self.error_per_epoch


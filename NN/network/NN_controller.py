
class NN_controller:

    def __init__(self, NeuralNetwork, error_type):
        self.NeuralNetwork = NeuralNetwork
        self.error_per_epoch = []
        self.accuracy_per_epoch = []
        self.error_type = error_type

    def train_epochs(self, input_set, exp_output_set, epochs):
        for i in range(epochs):
            error, accuracy = self.NeuralNetwork.train(input_set, exp_output_set)
            self.error_per_epoch.append(error)
            self.accuracy_per_epoch.append(accuracy)

        return self.error_per_epoch, self.accuracy_per_epoch

    def eval(self, input_set, exp_output_set):
        error, accuracy = self.NeuralNetwork.eval(input_set, exp_output_set)
        # self.error_per_epoch.append(error)
        # self.accuracy_per_epoch.append(accuracy)
        return error, accuracy


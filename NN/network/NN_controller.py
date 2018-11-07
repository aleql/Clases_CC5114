
class NN_controller:

    def __init__(self, NeuralNetwork, error_type):
        self.NeuralNetwork = NeuralNetwork
        self.error_per_epoch = []
        self.error_type = error_type

    def train_epochs(self, input_set, exp_output_set, epochs):
        for i in range(epochs):
            # self.error_per_epoch.append(self.NeuralNetwork.get_error())
            # self.error_per_epoch.append(self.NeuralNetwork.get_error())
            error, accuracy = self.NeuralNetwork.train(input_set, exp_output_set)
            self.error_per_epoch.append("Epoch {}, error : {}, accuracy: {}".format(i, error, accuracy))
        return self.error_per_epoch


from neuralnetwork.NeuronLayer import NeuronLayer

class NeuralNetwork(object):
    def __init__(self, input_number, hidden_layers=[], output_number=1, learning_rate=0.5):
        self.learning_rate = learning_rate
        hidden_layers.append(output_number)
        self.first_layer = NeuronLayer(input_number, hidden_layers[0], learning_rate)
        current_layer = self.first_layer
        for index in range(1, len(hidden_layers)):
            next_input_number = hidden_layers[index - 1]
            next_neurons_number = hidden_layers[index]
            current_layer = NeuronLayer(next_input_number, next_neurons_number, learning_rate, current_layer)
        self.last_layer = current_layer

    def feed(self, inputs):
        return self.first_layer.feed(inputs)

    def train(self, inputs, desired_output):
        self.feed(inputs)
        self.last_layer.back_propagate_error(desired_output)
        self.first_layer.update_weights(inputs)

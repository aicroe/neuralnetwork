from neuralnetwork.neuron.AbstractNeuron import AbstractNeuron

class Perceptron(AbstractNeuron):
    def __init__(self, bias, weights):
        AbstractNeuron.__init__(self, bias, weights)

    def feed(self, inputs):
        sumatory = 0
        for index in range(0, len(self.weights)):
            sumatory += self.weights[index] * inputs[index]
        output = sumatory + self.bias
        if output > 0:
            return 1
        else :
            return 0

    def train(self, inputs, desired_output):
        real_output = self.feed(inputs)
        difference = desired_output - real_output
        learning_rate = 0.1
        for index in range(0, len(self.weights)):
            old_weight = self.weights[index]
            current_input = inputs[index]
            new_weight = old_weight + (learning_rate * current_input * difference)
            self.weights[index] = new_weight
            self.bias = self.bias + (learning_rate * difference)

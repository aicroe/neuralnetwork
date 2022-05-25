from math import exp
from neuralnetwork.neuron.AbstractNeuron import AbstractNeuron

def transfer_derivate(val):
    return val * (1.0 - val)

class Neuron(AbstractNeuron):
    def __init__(self, bias, weights):
        AbstractNeuron.__init__(self, bias, weights)
        self.delta = None
        self.output = None

    def feed(self, inputs):
        sum = 0.0
        for index in range(0, len(self.weights)):
            sum += self.weights[index] * inputs[index]
        self.output = 1.0 / (1.0 + exp(-(sum + self.bias)))
        return self.output

    def train(self, inputs, desired_output):
        learning_rate = 0.5
        self.output = self.feed(inputs)
        error = desired_output - self.output
        delta = error * transfer_derivate(self.output)
        for index in range(0, len(self.weights)):
            self.weights[index] = self.weights[index] + (learning_rate * delta * inputs[index])
        self.bias = self.bias + (learning_rate * delta)

    def adjust_delta_with(self, error):
        self.delta = error * transfer_derivate(self.output)

    def adjust_bias_using_learning_rate(self, learning_rate):
        self.bias = self.bias + (learning_rate * self.delta)

    def adjust_weight_with_input(self, inputs, learning_rate):
        for index in range(0, len(inputs)):
            self.weights[index] = self.weights[index] + (learning_rate * self.delta * inputs[index])

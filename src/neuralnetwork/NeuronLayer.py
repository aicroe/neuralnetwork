from random import random
from neuralnetwork.neuron.SigmoidNeuron import Neuron

def generate_rand_bias():
    return random()

def generate_rand_weights(size):
    return [random() for _ in range(size)]

def create_neuron_with(inputs_length):
    return Neuron(generate_rand_bias(), generate_rand_weights(inputs_length))

class NeuronLayer(object):
    def __init__(self, inputs_length, neuron_number, learning_rate=0.5, previous_layer=None, next_layer=None):
        self.learning_rate = learning_rate
        self.neurons = [create_neuron_with(inputs_length) for _ in range(neuron_number)]
        self.set_previous_layer(previous_layer)
        self.set_next_layer(next_layer)

    def set_previous_layer(self, previous_layer):
        self.previous_layer = previous_layer
        if previous_layer is not None:
            previous_layer.next_layer = self

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer
        if next_layer is not None:
            next_layer.previous_layer = self

    def is_output_layer(self):
        return self.next_layer is None

    def feed(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.feed(inputs))
        if self.is_output_layer():
            return outputs
        else:
            return self.next_layer.feed(outputs)

    def back_propagate_error(self, desired_outputs):
        if self.is_output_layer():
            for index in range(0, len(self.neurons)):
                neuron = self.neurons[index]
                error = desired_outputs[index] - neuron.output
                neuron.adjust_delta_with(error)
        else:
            for index in range(0, len(self.neurons)):
                error = 0.0
                for next_neuron in self.next_layer.neurons:
                    next_weight = next_neuron.weights[index]
                    next_delta = next_neuron.delta
                    error = error + (next_weight * next_delta)
                self.neurons[index].adjust_delta_with(error)
        if self.previous_layer is not None:
            self.previous_layer.back_propagate_error(None)

    def update_weights(self, inputs):
        next_inputs = []
        for neuron in self.neurons:
            next_inputs.append(neuron.output)
            neuron.adjust_weight_with_input(inputs, self.learning_rate)
            neuron.adjust_bias_using_learning_rate(self.learning_rate)
        if not self.is_output_layer():
            self.next_layer.update_weights(next_inputs)

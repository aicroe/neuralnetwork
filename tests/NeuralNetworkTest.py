import random
from unittest import TestCase
from neuralnetwork.NeuralNetwork import NeuralNetwork

def threshold(val):
    if val >= 0.5:
        return 1
    else:
        return 0

def success(network_output, desired_output):
    return list(map(threshold, network_output)) == desired_output

class NeuralNetworkTest(TestCase):
    def test_xor(self):
        random.seed(0)
        network = NeuralNetwork(2, [3], 1)
        for _ in range(1000):
            network.train([0, 0], [0])
            network.train([0, 1], [1])
            network.train([1, 0], [1])
            network.train([1, 1], [0])
        self.assertTrue(success(network.feed([0, 0]), [0]))
        self.assertTrue(success(network.feed([0, 1]), [1]))
        self.assertTrue(success(network.feed([1, 0]), [1]))
        self.assertTrue(success(network.feed([1, 1]), [0]))

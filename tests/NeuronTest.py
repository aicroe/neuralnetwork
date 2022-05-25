from unittest import TestCase
from neuralnetwork.neuron.SigmoidNeuron import Neuron

class NeuronTest(TestCase):
    def test_or(self):
        neuron = Neuron(0.5, [-1, -1])
        for _ in range(100):
            neuron.train([0, 0], 0)
            neuron.train([0, 1], 1)
            neuron.train([1, 0], 1)
            neuron.train([1, 1], 1)
        self.assertLess(neuron.feed([0, 0]), 0.5)
        self.assertGreaterEqual(neuron.feed([0, 1]), 0.5)
        self.assertGreaterEqual(neuron.feed([1, 0]), 0.5)
        self.assertGreaterEqual(neuron.feed([1, 1]), 0.5)

    def test_and(self):
        neuron = Neuron(0.5, [-1, -1])
        for _ in range(100):
            neuron.train([0, 0], 0)
            neuron.train([0, 1], 0)
            neuron.train([1, 0], 0)
            neuron.train([1, 1], 1)
        self.assertLess(neuron.feed([0, 0]), 0.5)
        self.assertLess(neuron.feed([0, 1]), 0.5)
        self.assertLess(neuron.feed([1, 0]), 0.5)
        self.assertGreaterEqual(neuron.feed([1, 1]), 0.5)

    def test_nand(self):
        neuron = Neuron(0.5, [-1, -1])
        for _ in range(100):
            neuron.train([0, 0], 1)
            neuron.train([0, 1], 1)
            neuron.train([1, 0], 1)
            neuron.train([1, 1], 0)
        self.assertGreaterEqual(neuron.feed([0, 0]), 0.5)
        self.assertGreaterEqual(neuron.feed([0, 1]), 0.5)
        self.assertGreaterEqual(neuron.feed([1, 0]), 0.5)
        self.assertLess(neuron.feed([1, 1]), 0.5)

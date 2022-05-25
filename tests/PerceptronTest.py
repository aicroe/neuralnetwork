from unittest import TestCase
from neuralnetwork.neuron.Perceptron import Perceptron

class PerceptronTest(TestCase):
    def test_or(self):
        perceptron = Perceptron(0.5, [-1, -1])
        for _ in range(100):
            perceptron.train([0, 0], 0)
            perceptron.train([0, 1], 1)
            perceptron.train([1, 0], 1)
            perceptron.train([1, 1], 1)
        self.assertEqual(perceptron.feed([0, 0]), 0)
        self.assertEqual(perceptron.feed([0, 1]), 1)
        self.assertEqual(perceptron.feed([1, 0]), 1)
        self.assertEqual(perceptron.feed([1, 1]), 1)

    def test_and(self):
        perceptron = Perceptron(0.5, [-1, -1])
        for _ in range(100):
            perceptron.train([0, 0], 0)
            perceptron.train([0, 1], 0)
            perceptron.train([1, 0], 0)
            perceptron.train([1, 1], 1)
        self.assertEqual(perceptron.feed([0, 0]), 0)
        self.assertEqual(perceptron.feed([0, 1]), 0)
        self.assertEqual(perceptron.feed([1, 0]), 0)
        self.assertEqual(perceptron.feed([1, 1]), 1)

    def test_nand(self):
        perceptron = Perceptron(0.5, [-1, -1])
        for _ in range(100):
            perceptron.train([0, 0], 1)
            perceptron.train([0, 1], 1)
            perceptron.train([1, 0], 1)
            perceptron.train([1, 1], 0)
        self.assertEqual(perceptron.feed([0, 0]), 1)
        self.assertEqual(perceptron.feed([0, 1]), 1)
        self.assertEqual(perceptron.feed([1, 0]), 1)
        self.assertEqual(perceptron.feed([1, 1]), 0)

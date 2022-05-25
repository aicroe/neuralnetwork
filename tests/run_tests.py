import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from .NeuronTest import NeuronTest
from .PerceptronTest import PerceptronTest
from .NeuralNetworkTest import NeuralNetworkTest

test_loader = unittest.TestLoader()

suite = test_loader.loadTestsFromTestCase(NeuronTest)
suite.addTest(test_loader.loadTestsFromTestCase(PerceptronTest))
suite.addTest(test_loader.loadTestsFromTestCase(NeuralNetworkTest))

unittest.TextTestRunner(verbosity=2).run(suite)

class AbstractNeuron(object):
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights

    def feed(self, inputs):
        raise NotImplementedError

    def train(self, inputs, desired_output):
        raise NotImplementedError

import numpy as np


class Layer:
    """A layer of neural networks"""

    def __init__(self, num_inputs, num_nodes, bias):
        self.data = []
        self.targets = []
        self.num_inputs = num_inputs
        self.bias = bias
        #                 +1 for bias
        self.nodes = [Node(num_inputs + 1) for i in range(num_nodes)]

    def fit(self, data, targets):
        self.train(data, targets)

    def train(self, data, targets):
        self.data = data
        self.targets = targets

    def predict(self, data):
        # feed the data through
        outputs = []

        for i in range(len(self.nodes)):
            # get the output. append the bias as an input
            outputs.append(self.nodes[i].get_output(np.append(data, [self.bias])))

        return outputs


class Node:
    """A neural network node"""

    def __init__(self, num_inputs):
        # small random weights
        self.weights = np.random.ranf(num_inputs) - .5

    def get_output(self, inputs):
        total = 0

        # add up the weights
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]

        # the threshold is 0
        if total > 0:
            return 1
        else:
            return 0

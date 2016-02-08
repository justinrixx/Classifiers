import numpy as np


class Layer:
    """A layer of neural networks"""

    def __init__(self, num_inputs, num_nodes, bias):
        self.data = []
        self.targets = []
        self.std_devs = []
        self.means = []
        self.num_inputs = num_inputs
        self.bias = bias
        self.nodes = []

        # initialize the nodes
        for i in range(num_nodes):
            self.nodes.append(Node(num_inputs + 1))

    def fit(self, data, targets):
        self.train(data, targets)

    def train(self, data, targets):
        self.data = data
        self.targets = targets

        # scale all the data using z-scores
        for j in range(len(self.data[0])):
            # save the standard deviation and meancol = self.data[:, i]
            self.std_devs.append(np.std(self.data[:, j]))
            self.means.append(np.mean(self.data[:, j]))

            # scale the data
            self.data[:, j] -= self.means[j]
            self.data[:, j] /= self.std_devs[j]

    def predict(self, data):

        # scale the data
        for i, val in enumerate(data):
            data[i] -= self.means[i]
            data[i] /= self.std_devs[i]

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

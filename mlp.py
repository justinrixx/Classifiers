import numpy as np
import math


class MLP:
    """A multi-layer perceptron network"""

    def __init__(self, topology):
        self.data = []
        self.targets = []
        self.std_devs = []
        self.means = []
        self.num_inputs = -1
        self.outputs = []
        self.layers = []
        self.topology = topology

    def fit(self, data, targets):

        self.data = data
        self.targets = targets

        # normalize all the data
        for j in range(len(self.data[0])):
            # save the standard deviation and meancol = self.data[:, i]
            self.std_devs.append(np.std(self.data[:, j]))
            self.means.append(np.mean(self.data[:, j]))

            # scale the data
            self.data[:, j] -= self.means[j]
            self.data[:, j] /= self.std_devs[j]

        # determine how many inputs there are
        self.num_inputs = len(data[0])

        # find the possible outputs
        target_set = set(targets)
        self.outputs = list(target_set)

        # add the input and hidden layers
        for i in range(len(self.topology)):
            # first layer
            if i == 0:
                self.layers.append(Layer(self.num_inputs, self.topology[i]))
            else:
                self.layers.append(Layer(len(self.layers[i - 1].nodes), self.topology[i]))

        # add the output layer
        self.layers.append(Layer(len(self.layers[-1].nodes), len(self.outputs)))

    def predict(self, data):

        predictions = []

        for instance in data:

            # scale the data
            for i, val in enumerate(instance):
                instance[i] -= self.means[i]
                instance[i] /= self.std_devs[i]

            # run the instance through the network
            outputs = []
            for i in range(len(self.layers)):
                # first layer
                if i == 0:
                    outputs = self.layers[i].calc_outputs(instance)
                else:
                    outputs = self.layers[i].calc_outputs(outputs)

            # now decide on the correct one based on the outputs
            predictions.append(self.outputs[np.argmax(outputs)])

        return predictions


class Layer:
    """A layer of neural networks"""

    def __init__(self, num_inputs, num_nodes):
        self.num_inputs = num_inputs
        self.nodes = []
        self.outputs = []

        # initialize the nodes
        for i in range(num_nodes):
            self.nodes.append(Node(num_inputs))

    def calc_outputs(self, inputs):
        self.outputs = []
        for node in self.nodes:
            self.outputs.append(node.get_output(inputs))

        return self.outputs

    def get_outputs(self):
        return self.outputs


class Node:
    """A neural network node"""

    def __init__(self, num_inputs):
        # small random weights (+1 for bias)
        self.weights = np.random.ranf(num_inputs + 1) - .5

    def get_output(self, inputs):

        # append the bias
        inputs = np.append(inputs, np.array([-1]))

        total = 0

        # add up the weights
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]

        # do a sigmoid activation function
        return 1 / (1 + math.exp(-total))
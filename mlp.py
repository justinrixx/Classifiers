import numpy as np
import math


class MLP:
    """A multi-layer perceptron network"""

    def __init__(self, topology, eta, num_epochs):
        self.data = []
        self.targets = []
        self.std_devs = []
        self.means = []
        self.num_inputs = -1
        self.outputs = []
        self.layers = []
        self.topology = topology
        self.eta = eta
        self.num_right = 0
        self.num_wrong = 0
        self.num_epochs = num_epochs

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

        # how many epochs?
        for i in range(self.num_epochs):
            self.num_right = 0
            self.num_wrong = 0
            for j in range(len(self.data)):
                self.train(self.data[j], self.targets[j])

            # print for graphing purposes
            print(self.num_right / (self.num_right + self.num_wrong) * 100)

    def train(self, data, target):
        # feed the data through
        outputs = []
        for i in range(len(self.layers)):
            if i == 0:
                outputs = self.layers[i].calc_outputs(data)
            else:
                outputs = self.layers[i].calc_outputs(outputs)

        # The target is all zeros and one one
        targets = [0] * len(self.outputs)
        targets[self.outputs.index(target)] = 1
        if self.outputs[np.argmax(outputs)] == target:
            self.num_right += 1
        else:
            self.num_wrong += 1

        # BEGIN set new weights
        for i in reversed(range(len(self.layers))):

            error = 0
            outputs = self.layers[i].get_outputs()
            # calculate the error
            if i == len(self.layers) - 1:
                # output layer
                for j in range(len(self.layers[i].nodes)):
                    error = outputs[j] * (1 - outputs[j]) * (outputs[j] - targets[j])
                    self.layers[i].nodes[j].error = error
            else:
                # hidden layer
                for j in range(len(self.layers[i].nodes)):
                    for k in range(len(self.layers[i + 1].nodes)):
                        error += self.layers[i + 1].nodes[k].weights[j] * self.layers[i + 1].nodes[k].error
                    error *= outputs[j] * (1 - outputs[j])
                    self.layers[i].nodes[j].error = error

            # now compute the new weights
            for j in range(len(self.layers[i].nodes)):

                outputs = []
                if i == 0:
                    # input layer
                    outputs = data
                else:
                    outputs = self.layers[i - 1].get_outputs()

                # add a -1 for bias
                outputs = np.append(outputs, np.array([-1]))

                for k in range(len(self.layers[i].nodes[j].weights)):
                    self.layers[i].nodes[j].new_weights[k] = self.layers[i].nodes[j].weights[k] \
                                                    - self.eta * self.layers[i].nodes[j].error * outputs[k]

        # END set new weights

        # apply the new weights
        for layer in self.layers:
            layer.apply_new_weights()

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

    def apply_new_weights(self):
        for node in self.nodes:
            node.apply_new_weights()


class Node:
    """A neural network node"""

    def __init__(self, num_inputs):
        # small random weights (+1 for bias)
        self.weights = np.random.ranf(num_inputs + 1) - .5

        # used to store the new weights
        self.new_weights = np.random.ranf(num_inputs + 1) - .5

        # for error calculation
        self.error = 0

    def get_output(self, inputs):

        # append the bias at the end
        inputs = np.append(inputs, np.array([-1]))

        total = 0

        # add up the weights
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]

        # do a sigmoid activation function
        return 1 / (1 + math.exp(-total))

    def apply_new_weights(self):
        self.weights = np.copy(self.new_weights)
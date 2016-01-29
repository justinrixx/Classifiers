import copy
import numpy as np


def calculate_entropy(probabilities):
    """
    Calculates the entropy given a list of probabilities
    :param probabilities: A list of probabilities for which to
           calculate entropy
    """
    to_return = 0

    for probability in probabilities.values():
        if probability != 0:
            to_return += -probability * np.log2(probability);

    return to_return


def calculate_information_gain(s, f):
    """
    Calculates the information gain of using a particular attribute
    :param s: The entire data set
    :param f: The index of the feature to analyze
    :return: The information gain of S using a feature F (ok not really
             information gain. Rather than calculating the entropy of the
             entire set every time, I'm just returning the second part of
             the equation for information gain ( (|s_f| / |s|) * Entropy(s_f) )
    """

    # make sure we're looking at a valid column
    if f >= len(s[0]):
        return 0

    loss = 0

    # get a list of all possible values of the feature
    values = []
    for instance in s:
        if instance[f] not in values:
            values.append(instance[f])

    # calculate the loss
    for value in values:

        # get the sublist that has value for the feature at column f
        s_f = [x for x in s if x[f] == value]

        # get the frequency of each target
        frequencies = {}
        total = 0
        for instance in s_f:
            if instance[-1] in frequencies.keys():
                frequencies[instance[-1]] += 1
            else:
                frequencies[instance[-1]] = 1

            total += 1

        # turn frequency into probability
        for key in frequencies.keys():
            frequencies[key] = frequencies[key] / total

        # get the entropy
        entropy_s_f = calculate_entropy(frequencies)

        loss += (len(s_f) / len(s)) * entropy_s_f

    return loss


def build_tree(data, features):

    # check for empty data
    if len(data) == 0:
        # guess the first class
        return 0

    # check if all the targets are the same
    targets = [x[-1] for x in data]  # data[-1]
    test = np.full_like(targets, targets[0])
    if np.array_equal(targets, test):
        return targets[0]

    # check if there are any more features to test
    elif len(features) == 0:

        # choose the most common class
        unique, pos = np.unique(targets, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()

        return unique[maxpos]

    # calculate the information gain (or rather loss)
    else:
        # create a new node
        node = ID3_Node()

        losses = np.zeros(len(features))
        for feature in range(len(features)):
            losses[feature] = calculate_information_gain(data, feature)

        # go with the smallest
        i_smallest = np.argmin(losses)
        desired_feature = features[i_smallest]
        vals = [x[desired_feature] for x in data]
        all_values = set(vals)  # get all possible values of this feature

        node.attr_index = desired_feature

        # get the data and recurse
        for value in all_values:
            # get the parameters ready to pass down
            new_data = [x for x in data if x[desired_feature] == value]
            new_features = copy.deepcopy(features)
            new_features.remove(desired_feature)

            # recurse down the subtree
            subtree = build_tree(new_data, new_features)

            # once done, add the subtree into the main tree
            node.branches[value] = subtree

        # return once the subtrees are all in
        return node


class DTreeClassifier:
    """A decision tree classifier"""

    def __init__(self):
        self.data = ""
        self.targets = ""
        self.root = ""

    def fit(self, data, targets):
        self.train(data, targets)

    def train(self, data, targets):
        self.data = data
        self.targets = targets

        # zip the data and targets together
        targets = targets.reshape((-1, 1))
        datatargets = np.append(data, targets, axis=1)

        # features are a list from 0 to the size of the list
        features = []
        for i in range(self.data.shape[1]):
            features.append(i)

        self.root = build_tree(datatargets, features)

        self.output_tree(self.root, 0)

    def predict(self, data):

        toreturn = []

        for instance in data:

            toreturn.append(self.traverse_tree(instance, self.root))

        return toreturn

    def traverse_tree(self, instance, node):

        # check for a node vs a leaf
        if instance[node.attr_index] not in node.branches.keys():
            return 0

        branch = node.branches[instance[node.attr_index]]

        # see if more nodes exist or else it's a leaf
        if isinstance(branch, ID3_Node):
            return self.traverse_tree(instance, branch)
        else:
            return node.branches[instance[node.attr_index]]

    def output_tree(self, node, level):
        # create tabs
        tabs = ""
        i = 0
        while i < level:
            tabs += "\t"
            i += 1

        if isinstance(node, ID3_Node):
            print(tabs, "node attribute: ", node.attr_index)
            print(tabs, "children: ")
            for key, val in node.branches.items():
                print(tabs, key, ": ")
                self.output_tree(val, level + 1)
        else:
            print(tabs, node)


class ID3_Node:
    """
    This class is really just a struct to hold information
    about a node in a decision tree.
    """
    def __init__(self):
        self.attr_index = 0
        self.branches = {}

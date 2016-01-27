import copy
import numpy as np


def calculate_entropy(probabilities):
    """
    Calculates the entropy given a list of probabilities
    :param probabilities: A list of probabilities for which to
           calculate entropy
    """
    to_return = 0

    for probability in probabilities:
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
    if f >= s.shape()[1]:
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
        s_f = [x for x in s and x[f] == value]

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

        loss += (len(s_f) / s.shape()[0]) * entropy_s_f

    return loss


def get_all_classes(data, feature):
    classes = set()

    for instance in data:
        classes.add(instance)


def build_tree(data, features):
    # check if all the targets are the same
    targets = data[-1]
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
        losses = np.zeros(len(features))
        for feature in range(len(features)):
            losses[feature] = calculate_information_gain(data, feature)

        # go with the smallest
        i_smallest = np.argmin(losses)
        all_values = set(data[:, i_smallest])  # get all possible values of this feature

        tree = {features[i_smallest] : {}}

        # get the data and recurse
        for value in all_values:
            # get the parameters ready to pass down
            new_data = [x for x in data if x[i_smallest] == value]
            new_features = copy.deepcopy(features)
            new_features.remove(i_smallest)

            # recurse down the subtree
            subtree = build_tree(new_data, new_features)

            # once done, add the subtree into the main tree
            tree[features[i_smallest]][value] = subtree

        # return once the subtrees are all in
        return tree


class DTreeClassifier:
    """A decision tree classifier"""

    def __init__(self):
        self.data = ""
        self.targets = ""
        self.tree = ""

    def fit(self, data, targets):
        self.train(data, targets)

    def train(self, data, targets):
        self.data = data
        self.targets = targets

        # zip the data and targets together
        datatargets = np.concatenate((data, targets), axis=1)
        features = []
        for i in range(self.data.shape[1]):
            features.append(i)

        self.tree = build_tree(datatargets, features)


    def predict(self, data):

        toreturn = []

        for instance in data:

            var = 5

        return toreturn

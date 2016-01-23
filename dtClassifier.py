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


class DTreeClassifier:
    """A decision tree classifier"""

    def __init__(self):
        self.data = ""
        self.targets = ""

    def fit(self, data, targets):
        self.train(data, targets)

    def train(self, data, targets):
        self.data = data
        self.targets = targets


    def predict(self, data):

        toreturn = []

        for instance in data:

            var = 5

        return toreturn

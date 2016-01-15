import numpy as np
from collections import Counter


def euclidean_distance(instance1, instance2):
    """Calculates the distance from one instance to another"""

    distance = 0

    for i in range(len(instance1)):
        # positive numbers -- normal distance
        if instance1[i] >= 0:
            distance += (instance1[i] - instance2[i]) ** 2

        # negative numbers -- only add 1 if they don't match
        else:
            if instance1[i] != instance2[i]:
                distance += 1

    return distance


def most_common(lst):
    data = Counter(lst)
    return data.most_common()[0][0]


class KnnClassifier:
    """An implementation of the k nearest neighbor algorithm"""

    def __init__(self, k=1):
        self.std_devs = []
        self.means = []
        self.k = k
        self.data = ""
        self.targets = ""

    def fit(self, data, targets):
        self.train(data, targets)

    def train(self, data, targets):
        self.data = data
        self.targets = targets

        # scale all the data using z-scores
        for i in range(len(self.data[0])):
            # save the standard deviation and meancol = self.data[:, i]
            self.std_devs.append(np.std(self.data[:, i]))
            self.means.append(np.mean(self.data[:, i]))

            # scale the data
            self.data[:, i] -= self.means[i]
            self.data[:, i] /= self.std_devs[i]

    def predict(self, data):

        toreturn = []

        for guess in data:

            # store the distances
            distances = []

            # scale the data
            for i, val in enumerate(guess):
                guess[i] -= self.means[i]
                guess[i] /= self.std_devs[i]

            for instance in self.data:
                distances.append(euclidean_distance(instance, guess))

            # find the closest ones
            closest = np.argsort(distances)

            neighbors = []

            # check k neighbors
            for i in range(min(self.k, len(closest))):
                neighbors.append(self.targets[closest[i]])
                #for j, k in enumerate(closest):
                #    if k == i:
                #        neighbors.append(self.targets[j])
                #        continue

            toreturn.append(most_common(neighbors))

        return toreturn

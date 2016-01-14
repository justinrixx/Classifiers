import numpy as np
from collections import Counter


def euclidean_distance(instance1, instance2):

    distance = 0

    for i in range(len(instance1)):
        distance += (instance1[i] - instance2[i]) ** 2

    return distance


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


class KnnClassifier:
    """An implementation of the k nearest neighbor algorithm"""

    def __init__(self, k=1):
        self.k = k
        self.data = ""
        self.targets = ""

    def train(self, data, targets):
        self.data = data
        self.targets = targets

    def predict(self, guess):

        # store the distances
        distances = []

        for instance in self.data:
            distances.append(euclidean_distance(instance, guess))

        # find the closest ones
        closest = np.argsort(distances, axis=0)

        neighbors = []

        # check k neighbors
        for i in range(min(self.k, len(closest))):
            for j, k in enumerate(closest):
                if k == i:
                    neighbors.append(j)
                    continue

        choice = most_common(neighbors)

        return self.targets[choice]

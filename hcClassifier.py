class HcClassifier:
    """A really simple hardcoded classifier"""

    def __init__(self):
        self.choice = ""

    def train(self, data, targets):
        self.choice = targets[0]

    def predict(self, data):
        return self.choice

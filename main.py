import sys
import time  # for a random seed
import numpy as np
from sklearn import datasets
from hcClassifier import HcClassifier
from knnClassifier import KnnClassifier
from sklearn.neighbors import KNeighborsClassifier


def main(argv):

    # Handle command line arguments
    # 0. just use the default percent of the
    #   data set for training (70%)
    #
    # 1. should be a number from 0 to 1
    #   (0 <= x < 1) which determines what
    #   percent of the data set should be
    #   used for training
    #
    # 2. first argument is the filename of
    #   the file containing the data set
    tpercent = .7
    filename = ""

    if len(argv) == 2:
        tpercent = float(argv[1])
        # make sure it's in the legal range
        if tpercent < 0.0:
            tpercent *= -1
        while tpercent >= 1.0:
            tpercent -= 1.0

    elif len(argv) == 3:
        filename = argv[1]

        tpercent = float(argv[2])
        # make sure it's in the legal range
        if tpercent < 0.0:
            tpercent *= -1
        while tpercent >= 1.0:
            tpercent -= 1.0

    # get the time for use as a random seed
    #   this can be replaced by something else,
    #   but using the time will allow for a
    #   different shuffle each time
    seed = int(time.time())

    if filename == "":
        iris = datasets.load_iris()
        data = iris.data
        targets = iris.target

    # load from a file instead if that's the correct approach
    else:
        csv = np.genfromtxt(filename, delimiter=",")
        numcols = len(csv[0])
        data = csv[:, :numcols - 1]  # the first columns are the data
        targets = csv[:, numcols - 1]  # the last column is the targets

    # shuffle based on the time
    #   this uses the same seed (the time)
    #   in both the data and the targets so
    #   they match up after the shuffle
    np.random.seed(seed)
    np.random.shuffle(data)
    # reset the seed
    np.random.seed(seed)
    np.random.shuffle(targets)

    # determine the correct sizes of the sets
    tsize = int(tpercent * targets.size)
    psize = targets.size - tsize

    tdata = data[:tsize]
    pdata = data[tsize:tsize + psize]

    ttargets = targets[:tsize]
    ptargets = targets[tsize:tsize + psize]

    # train the classifier
    classifier = KnnClassifier(3)
    #classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(tdata, ttargets)

    # see how it did
    numcorrect = 0
    predictions = classifier.predict(pdata)
    for i in range(psize):
        if predictions[i] == ptargets[i]:
            numcorrect += 1

    percentcorrect = (numcorrect / psize) * 100.0

    print("Completed. Predicted", str(percentcorrect), "% correctly.")

# This is here to ensure main is only called when
#   this file is run, not just loaded
if __name__ == "__main__":
    main(sys.argv)

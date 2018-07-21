from __future__ import division
import numpy as np
from common import models, epsilons

epsilons = np.arange(0.01, 0.51, 0.01)
path = "attacks_{}/labels_e_{}.npy"

target_labels = np.load("mnist_labels.npy")
print("#epsilon\tnatural\tsecret")

for epsilon in epsilons:
    epsilon = round(epsilon, 2)
    accuracies = []
    del accuracies[:]
    for model in models:
        labels = np.load(path.format(model, epsilon))
        accuracies.append(1-np.count_nonzero(target_labels - labels)/np.prod(target_labels.shape))
    print("{}\t{}\t{}".format(epsilon, accuracies[0], accuracies[1]))
        

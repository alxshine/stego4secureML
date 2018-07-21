from __future__ import division
import numpy as np
from common import models, epsilons

samples =  np.load("mnist_samples.npy").reshape(10000,28,28)
path = "{}/e_{}.npy"
folder_path = "attacks_{}"
folders = [folder_path.format(model) for model in models]

print("%These are the changed pixels in percent for the different models")
print("%epsilon\tnatural\tsecret")

for epsilon in epsilons:
    adv_sample_sets = [np.load(path.format(folder, epsilon)).reshape(10000,28,28) for folder in folders]
    diffs = [np.count_nonzero(samples-adv_samples)/np.prod(samples.shape) for adv_samples in adv_sample_sets]
    print("{:03.2}\t{:17.16}\t{:17.16}".format(epsilon, diffs[0], diffs[1]))

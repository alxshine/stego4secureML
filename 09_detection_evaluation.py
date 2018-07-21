import numpy as np
from common import models, epsilons

threshold = 0.014321956650606307

labels = np.load("mnist_labels.npy")
diffs = np.load("mnist_diffs.npy")
folder_path = "attacks_{}"
folders = [folder_path.format(model) for model in models]

print("%the 'either-rate' (combined coverage of the model and our classifier) per model and epsilon")
print("%epsilon\tnatural\tadv_trained\tsecret")

for epsilon in epsilons:
    adv_label_sets = [np.load("{}/labels_e_{}.npy".format(folder, epsilon)) for folder in folders]
    adv_diff_sets = [np.load("{}/diffs_e_{}.npy".format(folder, epsilon)) for folder in folders]

    eithers = [(adv_label_sets[i] == labels) | (adv_diff_sets[i] > threshold) for i in range(2)]
    print("{:3}\t{:5}\t{:5}".format(epsilon, sum(eithers[0]), sum(eithers[1])))

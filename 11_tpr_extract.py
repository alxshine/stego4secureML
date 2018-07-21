from __future__ import division
import numpy as np
from common import models

folder_path = "attacks_{}"
label_path = "{}/labels_e_{}.npy"
diff_path = "{}/diffs_e_{}.npy"
threshold = 0.014321956650606307
#epsilons = np.arange(0.01, 0.51, 0.01)
epsilons=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.33,0.35,0.37,0.4,0.42,0.45,0.47,0.5]
labels = np.load("mnist_labels.npy")

print("%the TPR of the different models for all epsilons")
print("\t&\t")
for i in range(len(epsilons)):
    epsilon = epsilons[i]
    if i<len(epsilons)-1: 
        print("{:04.3f}\t&\t".format(epsilon)), 
    else:
        print("{:04.3f}\\\\".format(epsilon))

for model in models:
    print("{}\t&\t".format(model))
    folder = folder_path.format(model)
			
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        adv_labels = np.load(label_path.format(folder, epsilon))
        adv_diffs = np.load(diff_path.format(folder, epsilon))

        incorrect = labels != adv_labels
        flagged = adv_diffs> threshold

        tps = np.logical_and(incorrect, flagged)
        tpr = np.round(tps.sum()/10000,3)



        if i<len(epsilons)-1: 
            print("{:04.3f}\t&\t".format(tpr)),
        else:
            print("{:04.3f}\\\\".format(tpr))

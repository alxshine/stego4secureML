import numpy as np
import matplotlib.pyplot as plt
from common import models, epsilons, kernel, kernelResize, getKernelOffset, diffToEstimate, getWeightMatrix
import json

###############################################
#
# configuration
#
###############################################

folder = "attacks_{}"
path = "{}/{}e_{}.npy"
numRuns = 50
# used to count runs

###############################################
#
# benign samples
#
###############################################

# load benign samples stored as [0,255] ints
samples = np.load("mnist_samples.npy").astype(int).reshape((-1,28,28))

# load actual test labels
labels = np.load("mnist_labels.npy").astype(int)

# Benign difference mit min, max und mean
print("Calculating differences for benign samples")
benign_diffs = [diffToEstimate(sample, kernel, getWeightMatrix(sample)) for sample in samples]
print("Done")
benign_max = max(benign_diffs)
benign_min = min(benign_diffs)
benign_mean = np.mean(benign_diffs)
# decision_border = benign_max

np.save("mnist_diffs.npy", benign_diffs)

#extracted from training set:
#Training set differences: min 0.0003615763742060155, max 0.014321956650606307, mean nan
decision_border = 0.014321956650606307

###############################################
#
# adversarial samples for different epsilons
#
###############################################

for model in models:
    for epsilon in epsilons:
        print("Model {}, Epsilon: {}".format(model, epsilon))
        current_folder = folder.format(model)
        
        sample_path = path.format(current_folder, "", epsilon)
        predictions_path = path.format(current_folder, "labels_", epsilon)
        
        # load adversarial samples and convert them to [0,255] ints
        adv_samples = np.load(sample_path).reshape(10000, 28, 28) * 255
        adv_samples = adv_samples.astype(int)
        # load predictions
        predictions = np.load(predictions_path)
        
        # Adv difference with min, max and mean
        adv_diffs = np.zeros((adv_samples.shape[0]))
        # adv_diffs = [diffToEstimate(adv_sample, kernel, getWeightMatrix(adv_sample)) for adv_sample in adv_samples]
        for i in range(adv_samples.shape[0]):
            adv_diffs[i] = diffToEstimate(adv_samples[i], kernel, getWeightMatrix(adv_samples[i]))
        np.save("{}/diffs_e_{}.npy".format(current_folder, epsilon), adv_diffs)

#TODO: implement own logic
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from common import kernel, getWeightMatrix, diffToEstimate

x_train = input_data.read_data_sets("MNIST_data", one_hot=False).train.images.reshape((-1,28,28))

print("Calculating differences for training set")
index = 1
benign_diffs = []
for sample in x_train:
    print(index)
    benign_diffs.append(diffToEstimate(sample, kernel, getWeightMatrix(sample)))
    index += 1
    
print("Done")
benign_max = max(benign_diffs)
benign_min = min(benign_diffs)
benign_mean = np.mean(benign_diffs)

print("Training set differences: min {}, max {}, mean {}".format(benign_min, benign_max, benign_mean))

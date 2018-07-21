from tensorflow.examples.tutorials.mnist import input_data
from subprocess import run
import sys
import os
import numpy as np

print("Fetching models")
models = ['natural', 'secret']
for model in models:
    if os.path.isdir("models/{}".format(model)):
        print("Model {} already downloaded".format(model))
    else:
        fetch_args = ['python', 'mnist_challenge/fetch_model.py', model]
        completed = run(fetch_args)
        if completed.returncode != 0:
            print("The call to fetch_model for model {} returned an error".format(model))
            sys.exit(1)

#TODO: implement own logic
print("Saving mnist_data for reuse")
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
x_test = mnist.test.images
y_test = mnist.test.labels
np.save("mnist_samples.npy", x_test)
np.save("mnist_labels.npy", y_test)

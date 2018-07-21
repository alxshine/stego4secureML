import sys
import numpy as np
import matplotlib.pyplot as plt
from common import models, epsilons

folder_path = "attacks_{}"
folders = [folder_path.format(model) for model in models]
path = '{}/e_{}.npy'
estimate_path = '{}/estimates_e_{}.npy'
samples = np.load('mnist_samples.npy').reshape((-1,28,28))

try:
    index = int(sys.argv[1])
except IndexError:
    print("You can pass the image index as command line parameter")
    index = 42

plt.figure()

plt.subplot(221)
plt.imshow(samples[index])
plt.title("benign")

for i in range(2):
    folder = folders[i]
    plt.subplot(2,2,i+1)
    adv_samples = np.load(path.format(folder, 0.3)).reshape((10000,28,28))

    plt.imshow(adv_samples[index])
    plt.title(folder)

plt.show()

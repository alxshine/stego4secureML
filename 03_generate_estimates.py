from common import models, kernel, epsilons
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


for model in models:
    for epsilon in epsilons:
        adv_samples = np.load("attacks_{}/e_{}.npy".format(model, epsilon)).reshape(10000,28,28)
        adv_estimates = np.zeros_like(adv_samples)

        print("Generating estimates from adv_samples for epsilon {}".format(epsilon))
        for i in range(adv_samples.shape[0]):
            sample = adv_samples[i]
            estimate = convolve2d(adv_samples[i], kernel, mode='full')
            estimate = estimate[1:-1,1:-1]
            estimate -= estimate.min()
            estimate /= estimate.max()
            adv_estimates[i] = estimate

        adv_estimates = adv_estimates.reshape((10000,784))
        np.save("attacks_{}/estimates_e_{}.npy".format(model, epsilon), adv_estimates)

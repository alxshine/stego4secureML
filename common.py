import numpy as np
from scipy.signal import convolve2d

models = ['natural', 'secret']
epsilons = [round(e, 2) for e in np.arange(0.01, 0.51, 0.01)]

kernel = np.array([[-0.25, 0.5, -0.25],
                    [0.5, 0, 0.5],
                    [-0.25, 0.5, -0.25]])
mean_kernel = np.array([[0, 0.25, 0],
                       [0.25, 0, 0.25],
                       [0, 0.25, 0]])
large_mean_kernel = np.array([[0.125, 0.125, 0.125],
                       [0.125, 0, 0.125],
                       [0.125, 0.125, 0.125]])

def kernelResize(sample, kernel):
    offsetx, offsety = getKernelOffset(kernel)
    return sample[offsetx:sample.shape[0] - offsetx, offsety:sample.shape[1] - offsety]


def getKernelOffset(kernel):
    offsetx = int(kernel.shape[0] / 2)
    offsety = int(kernel.shape[1] / 2)
    return offsetx, offsety


def diffToEstimate(sample, kernel, weights=None):
    offsetx, offsety = getKernelOffset(kernel)
    estimated = convolve2d(sample, kernel, mode='valid')
    resized = kernelResize(sample, kernel)
    diff = resized - estimated
        
    if weights is None:
        weights = np.ones_like(resized) / np.prod(resized.shape)
        
    return np.mean(np.abs(np.multiply(diff, weights)))


def getWeightMatrix(sample):
    offsetx, offsety = getKernelOffset(large_mean_kernel)
    m = convolve2d(sample, mean_kernel, mode='valid')
    weights = np.zeros_like(kernelResize(sample, kernel))
    for x in range(weights.shape[0]):
        for y in range(weights.shape[1]):
            outerx = x + offsetx
            outery = y + offsety
            weights[x, y] = ((sample[outerx - 1, outery] - m[x, y]) ** 2 + (sample[outerx + 1, outery] - m[x, y]) ** 2
                + (sample[outerx, outery - 1] - m[x, y]) ** 2 + (sample[outerx, outery + 1] - m[x, y]) ** 2) / 3
    
    weights = np.divide(1., weights + 5)
    weights = weights / weights.sum()
    
    return weights



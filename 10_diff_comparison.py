import numpy as np
from common import models, epsilons

folder_path = "attacks_{}"
folders = [folder_path.format(model) for model in models]
path = '{}/e_{}.npy'
diff_path = '{}/diffs_e_{}.npy'

print("%The mins of the difference between the weighted estimates and the adv_samples")
print("%epsilon\tnatural\tsecret")

for epsilon in epsilons:
    diffs = [np.load(diff_path.format(folder, epsilon)) for folder in folders]
    means = [np.min(diff) for diff in diffs]

    print("{:03.2}\t{:17.16}\t{:17.16}".format(epsilon, means[0], means[1]))

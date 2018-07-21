import json
import numpy as np
from subprocess import run
import sys
import os
from common import models, epsilons

def cleanup():
    os.remove('config.json')

path = "attacks_{}/{}e_{}.npy"
pgd_args = ['python', 'mnist_challenge/pgd_attack.py']
run_args = ['python', 'mnist_challenge/run_attack.py']

with open('mnist_challenge/config.json', 'r+') as origFile:
    config = json.load(origFile)


for model in models:
    dir_path = "attacks_{}".format(model)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    print("Generating for model {}".format(model))
    for epsilon in epsilons:
        with open('config.json', 'w') as configFile:
            print("epsilon: {}".format(epsilon))
            config['epsilon'] = epsilon
            config['store_adv_path'] = path.format(model, "", epsilon)
            label_path = path.format(model, "labels_", epsilon)
            config['label_path'] = label_path
            config['model_dir'] = os.path.join('models', model)
            configFile.seek(0)
            json.dump(config, configFile, indent=1, sort_keys=False)
            configFile.truncate()
            
            #this is not very good style, but it is so much faster than copying the code from the original file and importing it
            # completed = run(pgd_args)
            # if completed.returncode != 0:
                # print("The call to pgd_attack.py for epsilon {} returned an error".format(epsilon), file=sys.stderr)
                # cleanup()
                # sys.exit(completed.returncode)

            completed = run(run_args)
            if completed.returncode != 0:
                print("The call to run_attack.py for epsilon {} returned an error".format(epsilon), file=sys.stderr)
                cleanup()
                sys.exit(completed.returncode)
            print("Moving pred.npy to {}".format(label_path))
            os.rename("pred.npy", label_path)

cleanup()

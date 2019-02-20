"""Generate a list of jobs to use across experiments, and save to a CSV.
"""

import random
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()


parser.add_argument('name', help='name for job table')
parser.add_argument('-s', '--seconds', type=int, default=100)
parser.add_argument('-m', '--models', type=int, default=2)
parser.add_argument('-c', '--containers', type=int, default=10)

args = parser.parse_args()

seconds = args.seconds
num_containers = args.containers
num_models = args.models


images = ['wzheng33/gru:latest', 'wzheng33/lstmcfc:latest', 'wzheng33/lstmcrf:latest', 'wzheng33/tc10:latest']
#images = ['mtynes/vae:latest', 'mtynes/mnist:latest']
images = random.sample(images, num_models)


def make_joblist(images, seconds, num_containers, name):
    images = np.random.choice(images, num_containers)
    seconds = np.sort(np.random.choice(list(range(seconds)), num_containers))
    table = pd.DataFrame.from_items(zip(['seconds', 'images'], [seconds, images]))
    table.to_csv('{}_jobtable.csv'.format(name), index=False)


make_joblist(images, seconds, num_containers, args.name)

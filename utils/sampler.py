import torch
from torch.utils.data import Sampler
from torch.nn.functional import softmax

from copy import deepcopy
import numpy as np
from numpy.random import choice

import math
import random
import itertools
from random import shuffle
import logging 
logger = logging.getLogger(__name__)

def UnbanlancedSample(index_per_class, frac, mode, weights, shuffle_weghts=True, **kwargs):
    """
    Create unbalanced dataset
    :param shuffle_weghts: whether shuffle weights of sampling or not
    :param args: if mode == 'gaussian' or mode == 'tail', mean and std should be passed in through args
    :return: list of sampled indexes of each class
    """
    num_classes = len(index_per_class)
    
    if weights == None:
        weights = []

    gaussian_fn = lambda x, mean, std: 1/(std*math.sqrt(2*math.pi)) * math.e**(((x-mean)/std)**2 /-2)
    if mode == "gaussian":
        # print(kwargs)
        # disturb based on the number of classes
        disturb = random.uniform(-1/num_classes, 1/num_classes)
        # magnitude of deviation
        x = kwargs["mean"] - (num_classes-1)/2 * kwargs["inr"] + disturb
        # generate weights
        weights = [gaussian_fn(x + i * kwargs["inr"], kwargs["mean"], kwargs["std"]) for i in range(num_classes)]

    elif mode == "tail":
        # disturb based on the number of classes
        disturb = random.uniform(-1/num_classes, 1/num_classes)
        # magnitude of deviation
        x = kwargs["mean"] + disturb
        # generate weights
        weights = [gaussian_fn(x + i * kwargs["inr"], kwargs["mean"], kwargs["std"]) for i in range(num_classes)]

    elif mode == "random":
        weights = [random.random() for i in range(num_classes)]

    elif mode == "equal":  # equal number
        weights = [frac] * num_classes
        
    # shuffle the weights
    if shuffle_weghts:
        random.shuffle(weights)

    sampled_indexes = []
    sampled_per_class = [0 for i in range(num_classes)]
    for i in range(num_classes):
        _size = round(len(index_per_class[i]) * weights[i])
        shuffle(index_per_class[i])
        sampled_indexes.append(index_per_class[i][:_size])
        sampled_per_class[i] = _size
    
    
    return sampled_indexes, sampled_per_class


def Splitdataset(mode, sampled_indexes, num_clients, replacement=True, inr=None):
    """
    Preare dataset for each client by spliting sampled dataset
    """
    indexes = deepcopy(sampled_indexes)
    dataset_per_client = [[] for i in range(num_clients)]  # list to store dataset of each client

    if mode == "equal_num":
        indexes = list(itertools.chain.from_iterable(indexes))
        num_per_client = len(indexes) // num_clients
        for i in range(num_clients):
            dataset_per_client[i] = np.random.choice(indexes, num_per_client, replace=replacement)

    elif mode == "random_num":

        indexes = list(itertools.chain.from_iterable(indexes))
        if inr is None:
            inr = 5

        num_per_client = len(indexes) // num_clients
        reserve_pool = []
        for i in range(num_clients):
            tmp_num = random.randint(-inr, inr)
            dataset_per_client[i] = choice(indexes, num_per_client + tmp_num, replace=replacement)
            reserve_pool.extend(dataset_per_client[i])
            indexes = np.setdiff1d(indexes, reserve_pool)

    elif mode == "equal_per_class":

        for i in range(num_clients):
            for j in range(len(indexes)):
                num_per_client = len(indexes[j]) // num_clients
                dataset_per_client[i].extend(
                    np.random.choice(indexes[j], num_per_client, replace=replacement)
                )

    return dataset_per_client


if __name__ == "__main__":
    from dataset import FLDataset
    dataset = FLDataset("/mnt/traffic/leijiachen/data", train=True, dataset="MNIST", num_classes=10)
    print(len(dataset))
    sampled_indexes = UnbanlancedSample(dataset.index_per_class, "random", replacement=True, shuffle_weghts=True)
    # dataset_per_client = Splitdataset("equal_num", sampled_indexes, 100, inr=30)
    # [print(len(item)) for item in dataset_per_client]
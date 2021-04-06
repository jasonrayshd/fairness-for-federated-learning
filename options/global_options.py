#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=30, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="wegith decay (default: 1e-4)")
    parser.add_argument('--optimizer', type=str, default='Adam', help="optimizer for updating")
    
    # other arguments
    parser.add_argument('--frac', type=float, default=1, help="the fraction of dataset")
    parser.add_argument('--root', type=str, default='/mnt/traffic/leijiachen/data', help="root of dataset")
    parser.add_argument('--mode', type=str, default='random', help="distribution of number of samples")
    parser.add_argument('--split_mode', type=str, default='equal_num', help="split sampled dataset for users, equal_num/random_num")
    parser.add_argument('--split_inr', type=int, default=10, help="turbulating range of the size of each user's dataset")
    parser.add_argument('--sample_inr', type=float, default=0.3, help="sample interval")
    parser.add_argument('--mean', type=float, default=0, help="mean of gaussian")
    parser.add_argument('--std', type=float, default=0.5, help="std of gaussian")
    parser.add_argument('--name', type=str, default='temp', help="name of the learning, which is used for logging and model saving")
    parser.add_argument('--model', type=str, default='CNNMnist', help="model to use")
    parser.add_argument('--dataset', type=str, default='MNIST', help="name of dataset")
    parser.add_argument('--num_channels', type=int, default=1, help="number of input channels")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--overwrite', action='store_true', help='overwrite log files or not')
    parser.add_argument('--split_replacement', type=bool, default=True, help='sample each client dataset with replacement or not')
    
    args = parser.parse_args()
    return args

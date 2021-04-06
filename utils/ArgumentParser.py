import argparse


def train_parser():
    
    parser = argparse.ArgumentParser(description="Parser for training")
    
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--name", type=str, default="temp")
    
    
    return parser.parse()
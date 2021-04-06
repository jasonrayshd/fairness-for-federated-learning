import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils import tensorboard

import ignite
from ignite import engine, metrics

from options import args_parser

import models

from update import FedAvg

from utils import FLDataset
from utils import init_logging, read_config

import os
import numpy as np
import shutil
import logging
from functools import partial

# ignore logging info from Engine
# logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def attach_metric_logger(
    evaluator: engine.Engine,
):
    @evaluator.on(engine.Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        metrics = evaluator.state.metrics

        message = ''
        for metric_name, metric_value in metrics.items():
            message += f'{metric_name}: {metric_value} '
        print(message)


def test():

    args = args_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dict = f"./checkpoints/{args.name}.pt"
    global_model = models.__dict__[args.model](args)
    global_model.load_state_dict(torch.load(model_dict))
    global_model = global_model.to(device)
    testset = FLDataset(path = args.root,
                        dataset = args.dataset,
                        train = False,
                        num_classes = args.num_classes,
                        transform = None,
                        target_transform=None)
    testloader = DataLoader(testset,
                            batch_size = args.bs,
                            )
    evaluator = engine.create_supervised_evaluator(
        model = global_model,
        device = device,
        non_blocking = True,
        metrics = {
            "Loss": metrics.Loss(nn.CrossEntropyLoss()),
            "Accuracy": metrics.Accuracy(),
            "ConfusionMatrix": metrics.ConfusionMatrix(args.num_classes),
            "IOU": metrics.IoU(metrics.ConfusionMatrix(args.num_classes))
        }
    )
    attach_metric_logger(evaluator)
    evaluator.run(testloader, max_epochs = 1)


if __name__ == "__main__":
    test()
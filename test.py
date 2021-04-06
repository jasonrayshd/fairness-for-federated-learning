import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torch.utils import tensorboard

import ignite
from ignite import engine, metrics
from ignite.utils import to_onehot

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
import seaborn as sns
import matplotlib.pyplot as plt


def attach_metric_logger(
    evaluator: engine.Engine,
    logger,
):
    @evaluator.on(engine.Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        metrics = evaluator.state.metrics

        message = ''
        for metric_name, metric_value in metrics.items():
            message += f'{metric_name}: {metric_value} '
            if metric_name == "ConfusionMatrix":
                getHeatmap(metric_value)
            
        logger.info(message)


def ot_per_class(output, index, num_classes):
    y_pred, y = output
    # probably, we have to apply torch.sigmoid if output is logits
    y_pred_bin = to_onehot(F.softmax(y_pred, dim=1).argmax(dim=1), num_classes=num_classes)
    y_ohe = to_onehot(y, num_classes=num_classes)
    return (y_pred_bin[:, index], y_ohe[:, index])

def getHeatmap(confusion_matrix):
    # initate seaborn and pyplot to draw confusion matrix heatmap
    sns.set()
    f,ax=plt.subplots()
    # draw heat map
    sns.heatmap(confusion_matrix,ax=ax)
    # set title and x/y label
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    
    plt.savefig(f"confusion_matrix.png")

def test():
    args = args_parser()

    logformat = "%(asctime)s - [%(filename)s:%(lineno)s-%(funcName)s()] - %(levelname)s - %(message)s"
    logging.basicConfig(filename=f"logs/{args.name}/test.log", format=logformat, level=logging.INFO)
    # logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)  # ignore logging info from Engine
    logger = logging.getLogger(__name__)

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
    # create evaluator
    evaluator = engine.create_supervised_evaluator(
        model = global_model,
        device = device,
        non_blocking = True,
        metrics = {
            "Loss": metrics.Loss(nn.CrossEntropyLoss()),
            "Accuracy": metrics.Accuracy(),
            "Precision": metrics.Precision(),
            "ConfusionMatrix": metrics.ConfusionMatrix(args.num_classes),
        }
    )
    # attach metrics logger
    acc_per_class = {}
    precision_per_class = {}
    for i in range(args.num_classes):
        acc_per_class["acc_{}".format(i)] = metrics.Accuracy(output_transform=partial(ot_per_class, index=i, num_classes=args.num_classes))
    for n, acc in acc_per_class.items():
        acc.attach(evaluator, name=n)

    attach_metric_logger(evaluator, logger)
    # start testing
    evaluator.run(testloader, max_epochs = 1)


if __name__ == "__main__":
    test()
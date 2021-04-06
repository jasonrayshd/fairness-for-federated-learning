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

from utils import attach_metric_logger, attach_training_logger, attach_model_checkpoint, attach_lr_scheduler
from utils import FLDataset, Splitdataset, UnbanlancedSample
from utils import init_logging, read_config

import os
import numpy as np
import shutil
import logging
from functools import partial

# ignore logging info from Engine
logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    # initialization
    args = args_parser()
    # config = read_config()  # for future expansion
    try:
        if args.overwrite:
            shutil.rmtree(f"./logs/{args.name}", ignore_errors=True)
            shutil.rmtree(f"./checkpoints/{args.name}", ignore_errors=True)
        os.mkdir(f"./logs/{args.name}")
    except:
        logger.info(f"log folder{args.name} exists.")

    init_logging(f"logs/{args.name}")
    logger.info(f"base settings:{args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device}")

    # datset preparation
    origin_dataset = FLDataset(
        path = args.root,
        dataset = args.dataset,
        train = True,
        num_classes = args.num_classes,
        transform = None,  # basic transform are normalization and totensor in FLDataset
        target_transform=None)

    index_per_class = origin_dataset.index_per_class
    weights = {
        "fixed_gaussian": [0.25, 0.35, 0.45, 0.6, 0.85, 0.85, 0.6, 0.45, 0.35, 0.25],
        "fixed_sharp_gaussian": [0.1, 0.3, 0.5, 0.7, 1, 1, 0.7, 0.5, 0.3, 0.1],
        "fixed_tail": np.linspace(0.1, 1, 10).tolist(),
        "gaussian": None,
        "tail": None,
        "random":None,
        "equal": None,
    }
    # sampled_indexes [...]
    kwargs =  {
            "mean" : args.mean,
            "std" : args.std, 
            "inr" : args.sample_inr
            }
    logger.info(f"total number of images in each category:{[len(index_per_class[i]) for i in range(10)]}")
    sampled_indexes, sampled_per_class = UnbanlancedSample(
        index_per_class = index_per_class,
        frac = args.frac,
        mode = args.mode,
        weights = weights[args.mode],
        shuffle_weghts = True,
        **kwargs
        )
    logger.info(f"Number of sampled data per category:{sampled_per_class}")
    logger.info(f"Total number of sampled dataset [{sum(sampled_per_class)}/{len(origin_dataset)}]")
    # clients_indexes [[...],[...],...]
    clients_indexes = Splitdataset(
        mode = args.split_mode,
        sampled_indexes = sampled_indexes,
        num_clients = args.num_users,
        replacement = args.split_replacement,
        inr = args.split_inr,
    )
    logger.info(f"Number of data in each class:{[len(item) for item in clients_indexes]}")
    for i, client_index in enumerate(clients_indexes):
        distribution = [0 for j in range(args.num_classes)]
        for j in range(len(client_index)):
            distribution[origin_dataset.labels[client_index[j]]] += 1
        logger.info(f"Distribution of client[{i}]'s dataset: {distribution}")

    # model preparation
    global_model = models.__dict__[args.model](args)
    global_state_dict = global_model.state_dict()

    for epoch in range(args.epochs):
        clients_model = [global_state_dict for i in range(args.num_users)]

        # logger.info(f"epoch:[{epoch+1}/{args.epochs}]")
        Updateclients(args,
                    epoch = epoch,
                    device = device,
                    global_model = global_model,
                    origin_dataset = origin_dataset,
                    clients_model = clients_model,
                    clients_indexes = clients_indexes)

        global_state_dict = FedAvg(clients_model)

    torch.save(global_state_dict, f"./checkpoints/{args.name}.pt")

def Updateclients(args, epoch, device, global_model, origin_dataset, clients_model, clients_indexes):

    loss_fn = nn.CrossEntropyLoss()

    for i, model in enumerate(clients_model):
        # logger.info(f"Client[{i+1}] start training.")
        clientSampler = torch.utils.data.SubsetRandomSampler(clients_indexes[i])
        dataloader = DataLoader(origin_dataset, batch_size=args.local_bs, sampler=clientSampler, drop_last=True)

        global_model.load_state_dict(model)
        global_model = global_model.to(device)

        _optimizer = partial(torch.optim.__dict__[args.optimizer],
                            global_model.parameters(),
                            lr = args.lr,
                            weight_decay = args.weight_decay)

        optimizer = None
        if args.optimizer == "SGD":
            optimizer = _optimizer(momentum = args.momentum)
        else:
            optimizer = _optimizer()

        trainer = engine.create_supervised_trainer(
            model = global_model,
            optimizer = optimizer,
            loss_fn = loss_fn,
            device = device,
            non_blocking = True,
        )


        writer = tensorboard.SummaryWriter(log_dir=f'summary/{args.name}')
        attach_training_logger(
                            trainer = trainer,
                            writer = writer,
                            info = {
                                "epoch_now": f"{epoch}/{args.epochs}",
                                "client_index": i,
                            },
                            log_interval = 5)

        logging.disable(logging.DEBUG)
        trainer.run(dataloader, max_epochs = args.local_ep)
        clients_model[i] = global_model.state_dict()

    evaluate(args, epoch, global_model, device, writer)


def evaluate(args, epoch, global_model, device, writer):
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
        }
    )
    attach_metric_logger(
        evaluator = evaluator,
        data_name = args.name,
        info = {
            "epoch_now": f"{epoch}/{args.epochs}",
        },
        writer = writer)
    evaluator.run(testloader, max_epochs = 1)


if __name__ == "__main__":
    main()
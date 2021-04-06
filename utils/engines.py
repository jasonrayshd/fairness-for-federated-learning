import logging
from typing import Dict

import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils import tensorboard
from ignite import engine
from ignite import handlers
from ignite.contrib import handlers as contrib_handlers


logger = logging.getLogger(__name__)

def attach_lr_scheduler(
    trainer: engine.Engine,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    writer: tensorboard.SummaryWriter,
):
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def update_lr(engine: engine.Engine):
        current_lr = lr_scheduler.get_last_lr()[0]
        logger.info(f'epoch: {engine.state.epoch} - current lr: {current_lr}')
        writer.add_scalar('learning_rate', current_lr, engine.state.epoch)

        lr_scheduler.step()


def attach_training_logger(
    trainer: engine.Engine,
    writer: tensorboard.SummaryWriter,
    info: Dict,
    log_interval: int = 10,
):
    @trainer.on(engine.Events.COMPLETED )
    def log_training_loss(engine: engine.Engine):
        epoch_length = engine.state.epoch_length
        epoch = engine.state.epoch
        output = engine.state.output

        # idx = engine.state.iteration
        # idx_in_epoch = (engine.state.iteration - 1) % epoch_length + 1

        # if idx_in_epoch % log_interval != 0:
        #     return

        logger.info(f'epoch now:[{info["epoch_now"]}]client[{info["client_index"]}] - epoch[{epoch}] loss: {output:.3f}')
        writer.add_scalar('loss', output)


def attach_metric_logger(
    evaluator: engine.Engine,
    data_name: str,
    info: Dict,
    writer: tensorboard.SummaryWriter,
):
    @evaluator.on(engine.Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        metrics = evaluator.state.metrics

        message = f'epoch now:[{info["epoch_now"]}] '
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f'{data_name}/{metric_name}', metric_value, engine.state.epoch)
            message += f'{metric_name}: {metric_value:.3f} '

        logger.info(message)


def attach_model_checkpoint(trainer: engine.Engine, models: Dict[str, nn.Module], prefix:str, name: str):
    def to_epoch(trainer: engine.Engine, event_name: str):
        return trainer.state.epoch

    handler = handlers.ModelCheckpoint(
        f'./checkpoints/{name}',
        f'{prefix}',
        create_dir=True,
        require_empty=False,
        n_saved=None,
        global_step_transform=to_epoch,
    )

    trainer.add_event_handler(engine.Events.COMPLETED, handler, models)

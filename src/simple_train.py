import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import ray
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback, TuneReportCheckpointCallback)
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest import Repeater
from ray.tune.suggest.hyperopt import HyperOptSearch

import utilities as ut
from net import IDENet

model_name = "resnet50"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
my_label = "4channel_predict" + '_' + model_name
seed_everything(2022)

# data_dir = "../datasets/NA12878_PacBio_MtSinai/"
data_dir = "../data"

if len(sys.argv) < 2:
    print("usage error!")
    exit()

hight = 224

logger = TensorBoardLogger(os.path.join(data_dir, "channel_predict"), name=my_label)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints_predict/" + my_label,
    filename='{epoch:02d}-{validation_mean:.2f}-{train_mean:.2f}',
    monitor="validation_mean",
    verbose=False,
    save_last=None,
    save_top_k=1,
    mode="max",
    auto_insert_metric_name=True,
    every_n_train_steps=None,
    train_time_interval=None,
    every_n_epochs=None,
    save_on_train_epoch_end=None,
    every_n_val_epochs=None
)

def main_train():
    config = {
        "lr": 7.1873e-06,
        "batch_size": 118, # 14,
        "beta1": 0.9,
        "beta2": 0.999,
        'weight_decay': 0.0011615,
        'model': sys.argv[1]
    }

    model = IDENet(data_dir, config)

    trainer = pl.Trainer(
        max_epochs=30,
        gpus=1,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)


def train_tune(config, checkpoint_dir=None, num_epochs=200, num_gpus=1):
    model = IDENet(data_dir, config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)

class MyStopper(tune.Stopper):
    def __init__(self, metric, value, epoch = 1):
        self._metric = metric
        self._value = value
        self._epoch = epoch

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        return (result["training_iteration"] > self._epoch) and (result[self._metric] < self._value)


    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return False

def gan_tune(num_samples=-1, num_epochs=40, gpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-8, 1e-4),
        "batch_size": 64,
        "beta1": 0.9, 
        "beta2": 0.999, 
        'weight_decay': tune.uniform(0, 0.01),
        'model': sys.argv[1]
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
        )

    reporter = CLIReporter(
        metric_columns=['train_loss', "train_mean", 'validation_loss', "validation_mean"]
        )

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
        ),
        local_dir=data_dir,
        resources_per_trial={
            "cpu": 4,
            "gpu": 1,
        },
        config=config,
        num_samples=num_samples,
        metric='validation_mean',
        mode='max',
        scheduler=scheduler,
        progress_reporter=reporter,
        resume="AUTO",
        max_failures = -1,
        name="tune" + model_name)

ray.init()
gan_tune()

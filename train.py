import utilities as ut
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count
import pysam
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import list2img
from hyperopt import hp

seed_everything(2022)

bam_data_dir = "../data/"
vcf_data_dir = "../data/"
data_dir = "../data/"
bs = 32
my_label = "ResNet34"

bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"

ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"

# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

logger = TensorBoardLogger(os.path.join(
    data_dir, "channel_predict"), name=my_label)

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
        "lr": 7.1873e-6,
        "batch_size": 14,
        "beta1": 0.9,
        "beta2": 0.999,
        'weight_decay': 0.0011615,
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
    
    for name, param in model.named_parameters():
        if "conv2ds" in name:
            param.requires_grad = False

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=16
    )
    trainer.fit(model)

class MyStopper(tune.Stopper):
    def __init__(self, metric, value, epoch=1):
        self._metric = metric
        self._value = value
        self._epoch = epoch

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        return (result["training_iteration"] > self._epoch) and (result[self._metric] < self._value)

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return False

def gan_tune(num_samples=-1, num_epochs=50, gpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-7, 1e-3),
        "batch_size": bs,
        "beta1": 0.9, 
        "beta2": 0.999, 
        'weight_decay': tune.uniform(0, 1e-4),
        'model': "resnet50"

    }

    bayesopt = HyperOptSearch(config, metric="validation_mean", mode="max")
    re_search_alg = Repeater(bayesopt, repeat=1)

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        metric_columns=['train_loss', "train_mean",
                        'validation_loss', "validation_mean"]
    )

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
        ),
        local_dir=data_dir,
        resources_per_trial={
            "cpu": 1,
            "gpu": 1,
        },
        num_samples=num_samples,
        metric='validation_mean',
        mode='max',
        scheduler=scheduler,
        progress_reporter=reporter,
        resume=False,
        search_alg=re_search_alg,
        max_failures=-1,
        name="tune")

ray.init()
gan_tune()

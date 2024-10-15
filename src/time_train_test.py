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
from hyperopt import hp
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Callback

seed_everything(2022)

data_dir = "/home/xzy/Desktop/MSVF-test/data/"
bs = 128
my_label = "mobilenet_v2"

bam_path = data_dir + "sorted_final_merged.bam"

ins_vcf_filename = data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = data_dir + "delete_result_data.csv.vcf"

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

class MyLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.epoch_times = []
        self.memory_usage = []

    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, trainer, pl_module):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        memory_allocated = torch.cuda.memory_allocated() / 1024.0 / 1024.0
        self.memory_usage.append(memory_allocated)

    def on_train_end(self, trainer, pl_module):
        total_time = sum(self.epoch_times)
        avg_time_per_epoch = total_time / len(self.epoch_times)

        with open('CSV-Filter_mobilenet_v2_time_memory_statistics.txt', 'w') as f:
            f.write('Total running time: {} seconds\n'.format(total_time))
            f.write('Time for each epoch: {}\n'.format(self.epoch_times))
            f.write('Average time per epoch: {} seconds\n'.format(avg_time_per_epoch))
            f.write('GPU memory usage for each epoch: {} MB\n'.format(self.memory_usage))

def main_train(checkpoint_dir=None, num_epochs=50, num_gpus=1):
    config = {
        "lr": 7.1873e-6,
        "batch_size": bs,
        "beta1": 0.9,
        "beta2": 0.999,
        'weight_decay': 0.0011615,
        'model': "resnet50"
    }

    model = IDENet(data_dir, config)
    my_logging_callback = MyLoggingCallback()

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        check_val_every_n_epoch=1,
        logger=False,
        callbacks=[checkpoint_callback, my_logging_callback],
        precision=16
    )

    trainer.fit(model)

start_time = time.time()
main_train()
end_time = time.time()
print('Total running time: {} seconds'.format(end_time - start_time))
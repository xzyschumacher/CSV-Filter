import utilities as ut
from pudb import set_trace
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

# data_dir = "../datasets/NA12878_PacBio_MtSinai/"
data_dir = "/home/xwm/DeepSVFilter/datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"

# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()
hight = 224
all_ins = torch.empty(0, 5, hight, hight)
all_del = torch.empty(0, 5, hight, hight)
all_n = torch.empty(0, 5, hight, hight)

for chromosome, chr_len in zip(chr_list, chr_length):
    print(chromosome)
    # ins = torch.cat((torch.load(data_dir + 'image/' + chromosome + '/ins_cigar_new_img' + '.pt'), torch.load(data_dir + 'image/' + chromosome + '/ins_img' + '.pt')[:, 2:3, :, :]), 1)
    # _del = torch.cat((torch.load(data_dir + 'image/' + chromosome + '/del_cigar_new_img' + '.pt'), torch.load(data_dir + 'image/' + chromosome + '/del_img' + '.pt')[:, 2:3, :, :]), 1)
    # n = torch.cat((torch.load(data_dir + 'image/' + chromosome + '/negative_cigar_new_img' + '.pt'), torch.load(data_dir + 'image/' + chromosome + '/negative_img' + '.pt')[:, 2:3, :, :]), 1)

    ins = torch.load(data_dir + 'image/' + chromosome + '/ins_cigar_new_img' + '.pt')
    _del = torch.load(data_dir + 'image/' + chromosome + '/del_cigar_new_img' + '.pt')
    n = torch.load(data_dir + 'image/' + chromosome + '/negative_cigar_new_img' + '.pt')

    all_ins = torch.cat((all_ins, ins), 0)
    all_del = torch.cat((all_del, _del), 0)
    all_n = torch.cat((all_n, n), 0)

torch.save(all_ins, 'all_ins_img' + '.pt')
torch.save(all_del, 'all_del_img' + '.pt')
torch.save(all_n, 'all_n_img' + '.pt')
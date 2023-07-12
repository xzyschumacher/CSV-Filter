import argparse
import os
import random
import subprocess

import numpy as np
import pandas as pd
import pysam
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import utilities as ut


def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    help = "The path of address"
    parser.add_argument('--thread_num', help=help)
    args = parser.parse_args()
    return args


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
seed_everything(2022)


data_dir = "../data/"

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"


sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

data_list = []
for chromosome, chr_len in zip(chr_list, chr_length):
    # if not os.path.exists(data_dir + 'flag/' + chromosome + '.txt'):
    data_list.append((chromosome, chr_len))

args = parse_args()
thread_num = int(args.thread_num)

for chr, length in data_list:
    num = len(subprocess.getoutput("ps -aux | grep process_file.py").split('\n'))
    while num > thread_num:
        num = len(subprocess.getoutput(
            "ps -aux | grep process_file.py").split('\n'))
        # print(num)
    print("python process_file.py --chr " + chr + " --len " + str(length))
    # subprocess.call("python create_process_file.py --chr " + chr + " --len " + str(len), shell = True)
    # fd = open(chr + ".txt")
    subprocess.Popen("python process_file.py --chr " +
                     chr + " --len " + str(length), shell=True)
    # subprocess.Popen("python par.py --chr " + chr + " --len " + str(len), shell=True)
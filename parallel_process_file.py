import argparse
import os
import random
import subprocess
import concurrent.futures

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
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    help = "The path of address"
    parser.add_argument('--thread_num', help=help)
    args = parser.parse_args()
    return args

def process_chromosome(chr, length, cpu_core):
    command = f"taskset -c {cpu_core} python process_file.py --chr {chr} --len {length}"
    print(command)
    subprocess.Popen(command, shell=True).wait()

seed_everything(2022)

bam_data_dir = "../data/"
vcf_data_dir = "../data/"

bam_path = bam_data_dir + "HG002-PacBio-CLR-minimap2.sorted.bam"
vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"


sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list_sam_file = sam_file.references
chr_length_sam_file = sam_file.lengths
sam_file.close()

allowed_chromosomes = set(f"{i}" for i in range(1, 23)) | {"X", "Y"}

chr_list = []
chr_length = []

for chrom, length in zip(chr_list_sam_file, chr_length_sam_file):
    if chrom in allowed_chromosomes:
        chr_list.append(chrom)
        chr_length.append(length)

hight = 224

data_list = []
for chromosome, chr_len in zip(chr_list, chr_length):
    data_list.append((chromosome, chr_len))

args = parse_args()
thread_num = int(args.thread_num)

def worker(chromosome_data, cpu_core):
    chr, length = chromosome_data
    process_chromosome(chr, length, cpu_core)

with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
    futures = []
    for i, chromosome_data in enumerate(data_list):
        core_index = i % thread_num 
        futures.append(executor.submit(worker, chromosome_data, core_index))
    concurrent.futures.wait(futures)

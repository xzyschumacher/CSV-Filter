
import sys
import os
from glob import glob

import numpy as np
import torch


def is_left_soft_clipped_read(read):
    if(read.cigartuples[0][0]==4):
        return True
    else:
        return False

def is_right_soft_clipped_read(read):
    if(read.cigartuples[-1][0]==4):
        return True
    else:
        return False

def draw_insertion(bam_path, chromosome, pic_length, data_dir):

    ref_chromosome_filename = data_dir + "chr/" + chromosome + ".fa"
    # fa = pysam.FastaFile(ref_chromosome_filename)
    # chr_string = fa.fetch(chromosome)
    sam_file = pysam.AlignmentFile(bam_path, "rb")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    split_read_left = torch.zeros(pic_length, dtype=torch.int)
    split_read_right = torch.zeros(pic_length, dtype=torch.int)
    rd_count = torch.zeros(pic_length, dtype=torch.int)
    conjugate_m = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i = torch.zeros(pic_length, dtype=torch.int)
    conjugate_d = torch.zeros(pic_length, dtype=torch.int)
    conjugate_s = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i_list = [[0] for _ in range(pic_length)]
    # match_count = torch.zeros(pic_length, dtype=torch.int)
    # mismatch_count = torch.zeros(pic_length, dtype=torch.int)
    # bam_op_count = torch.zeros([9, pic_length], dtype=torch.int)

    for read in sam_file.fetch(chromosome):
        if read.is_unmapped or read.is_secondary:
            continue
        start, end = (read.reference_start, read.reference_end)

        if is_left_soft_clipped_read(read):
            split_read_left[start:end] += 1

        if is_right_soft_clipped_read(read):
            split_read_right[start:end] += 1

        # ref_read = chr_string[start:end]

        # read = read.get_forward_sequence()

        # ref_read = chr_string[start:end]

        # read = read.get_forward_sequence()

        reference_index = start # % 2 == 0 :1  % 2 == 1 :0
        for operation, length in read.cigar: # (operation, length)
            if operation == 3 or operation == 7 or operation == 8:
                reference_index += length
            elif operation == 0:
                conjugate_m[reference_index:reference_index + length] += 1
                reference_index += length
            elif operation == 1:
                conjugate_i[reference_index] += length
                conjugate_i_list[reference_index].append(length)
            elif operation == 4:
                conjugate_s[reference_index - int(length / 2):reference_index + int(length / 2)] += 1
            elif operation == 2:
                conjugate_d[reference_index:reference_index + length] += 1
                reference_index += length

    sam_file.close()


    with open(data_dir + "depth/" + chromosome, "r") as f:
        for line in f:
            pos_count = line[:-1].split("\t")[1:]
            # set_trace()
            rd_count[int(pos_count[0]) - 1] = int(pos_count[1])

    # rd_count = MaxMinNormalization(rd_count)  # The scope of rd_count value is [0, 1]

    return torch.cat([split_read_left.unsqueeze(0), split_read_right.unsqueeze(0), rd_count.unsqueeze(0)],0), torch.cat([conjugate_m.unsqueeze(0), conjugate_i.unsqueeze(0), conjugate_d.unsqueeze(0), conjugate_s.unsqueeze(0)], 0), conjugate_i_list


def trans2img(bam_path, chromosome, chr_len, data_dir):

    #mean_size, std_size = estimateInsertSizes(bam_path, alignments=1000000)

    print("[*] Start generating images ===")
    chromosome_sign = draw_insertion(bam_path, chromosome, chr_len, data_dir)
    print("[*] End generating images ===")
    return chromosome_sign

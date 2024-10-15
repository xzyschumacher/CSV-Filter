import argparse
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import utilities as ut

torch.multiprocessing.set_sharing_strategy('file_system')

seed_everything(2022)

bam_data_dir = "/media/xzy/HDD_4T_2/DATASETS/Alignment_data/PacBio-CLR/"
vcf_data_dir = "../data/"
data_dir = "../data/"

bam_path = bam_data_dir + "HG002-PacBio-CLR-minimap2.sorted.bam"
# bam_path = data_dir + "NA12878_S1.bam"
# bam_path = data_dir + "NA12878_NGMLR_sorted.bam"

ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"

hight = 224
resize = torchvision.transforms.Resize([hight, hight])


def position(sum_data):
    chromosome, chr_len = sum_data
    ins_position = []
    del_position = []
    n_position = []
    # insert
    insert_result_data = pd.read_csv(ins_vcf_filename, sep="\t", index_col=0)
    insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
    row_pos = []
    for index, row in insert_chromosome.iterrows():
        row_pos.append(row["POS"])

    set_pos = set()

    for pos in row_pos:
        set_pos.update(range(pos - 1024, pos + 1024))

    for pos in row_pos:
        gap = 1024
        # positive
        begin = pos - 1 - gap
        end = pos - 1 + gap
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1

        ins_position.append([begin, end])

    # delete
    delete_result_data = pd.read_csv(del_vcf_filename, sep="\t", index_col=0)
    delete_chromosome = delete_result_data[delete_result_data["CHROM"] == chromosome]
    row_pos = []
    row_end = []
    for index, row in delete_chromosome.iterrows():
        row_pos.append(row["POS"])
        row_end.append(row["END"])

    for pos in row_pos:
        set_pos.update(range(pos - 1024, pos + 1024))

    for pos, end in zip(row_pos, row_end):
        gap = int((end - pos) / 4)
        if gap == 0:
            gap = 1
        # positive
        begin = pos - 1 - gap
        end = end - 1 + gap
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1

        del_position.append([begin, end])

        # negative
        del_length = end - begin

        for _ in range(2):
            end = begin

            while end - begin < del_length / 2 + 1:
                random_begin = random.randint(1, chr_len)
                while random_begin in set_pos:
                    random_begin = random.randint(1, chr_len)
                begin = random_begin - 1 - gap
                end = begin + del_length
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

            n_position.append([begin, end])

    save_path = data_dir + 'position/' + chromosome
    ut.mymkdir(save_path)
    torch.save(ins_position, save_path + '/insert' + '.pt')
    torch.save(del_position, save_path + '/delete' + '.pt')
    torch.save(n_position, save_path + '/negative' + '.pt')


def create_image(sum_data):
    chromosome, chr_len = sum_data

    print("deal " + chromosome)

    ins_position = torch.load(
        data_dir + 'position/' + chromosome + '/insert' + '.pt')
    del_position = torch.load(
        data_dir + 'position/' + chromosome + '/delete' + '.pt')
    n_position = torch.load(data_dir + 'position/' +
                            chromosome + '/negative' + '.pt')

    print(chromosome + " cigar start")
    save_path = data_dir + 'image/' + chromosome

    if os.path.exists(save_path + '/negative_cigar_new_img' + '.pt'):
        return

    ins_cigar_img = torch.empty(len(ins_position), 1, hight, hight)
    del_cigar_img = torch.empty(len(del_position), 1, hight, hight)
    negative_cigar_img = torch.empty(len(n_position), 1, hight, hight)

    for i, b_e in enumerate(ins_position):
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                ins_cigar_img[i] = ut.cigar_new_img_single_optimal(
                    bam_path, chromosome, b_e[0], b_e[1], zoom)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print("Exception cigar_img_single_optimal(ins_position) " + chromosome + " " + str(zoom) + ". The length = ", b_e[1] - b_e[0])
                           
        print("===== finish_cigar_img(ins_position) " + chromosome + " index = " + str(i) + "/" + str(len(ins_position)))

    for i, b_e in enumerate(del_position):
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                del_cigar_img[i] = ut.cigar_new_img_single_optimal(
                    bam_path, chromosome, b_e[0], b_e[1], zoom)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print("Exception cigar_img_single_optimal(del_position) " + chromosome + " " + str(zoom) + ". The length = ", b_e[1] - b_e[0])
            
        print("===== finish_cigar_img(del_position) " + chromosome + " index = " + str(i) + "/" + str(len(del_position)))

    for i, b_e in enumerate(n_position):
        zoom = 1

        fail = 1
        while fail:
            try:
                fail = 0
                negative_cigar_img[i] = ut.cigar_new_img_single_optimal(
                    bam_path, chromosome, b_e[0], b_e[1], zoom)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print("Exception cigar_img_single_optimal(neg_position) " + chromosome + " " + str(zoom) + ". The length = ", b_e[1] - b_e[0])


        print("===== finish_cigar_img(neg_position) " + chromosome + " index = " + str(i) + "/" + str(len(n_position)))
        
    ut.mymkdir(save_path)
    torch.save(ins_cigar_img, save_path + '/ins_cigar_new_img' + '.pt')
    torch.save(del_cigar_img, save_path + '/del_cigar_new_img' + '.pt')
    torch.save(negative_cigar_img, save_path +
               '/negative_cigar_new_img' + '.pt')
    print(chromosome + " cigar end")


def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    help = "The path of address"
    parser.add_argument('--chr', help=help)
    parser.add_argument('--len', help=help)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    position([args.chr, int(args.len)])
    create_image([args.chr, int(args.len)])

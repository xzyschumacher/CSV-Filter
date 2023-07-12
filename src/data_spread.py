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

data_dir = "../data/"
bam_path = data_dir + "sorted_final_merged.bam"

ins_vcf_filename = data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = data_dir + "delete_result_data.csv.vcf"

all_enforcement_refresh = 0
position_enforcement_refresh = 0
img_enforcement_refresh = 0
sign_enforcement_refresh = 0  # attention
cigar_enforcement_refresh = 0

# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

if os.path.exists(data_dir + '/all_n_img' + '.pt'):
    pass
else:
    all_ins_cigar_img = torch.empty(0, 1, hight, hight)
    all_del_cigar_img = torch.empty(0, 1, hight, hight)
    all_negative_cigar_img = torch.empty(0, 1, hight, hight)

    for chromosome, chr_len in zip(chr_list, chr_length):
        print("======= deal " + chromosome + " =======")

        print("position start")
        if os.path.exists(data_dir + 'position/' + chromosome + '/negative' + '.pt') and not position_enforcement_refresh:
            print("loading")
            ins_position = torch.load(
                data_dir + 'position/' + chromosome + '/insert' + '.pt')
            del_position = torch.load(
                data_dir + 'position/' + chromosome + '/delete' + '.pt')
            n_position = torch.load(
                data_dir + 'position/' + chromosome + '/negative' + '.pt')
        else:
            ins_position = []
            del_position = []
            n_position = []
            # insert
            insert_result_data = pd.read_csv(
                ins_vcf_filename, sep="\t", index_col=0)
            insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
            row_pos = []
            for index, row in insert_chromosome.iterrows():
                row_pos.append(row["POS"])

            set_pos = set()

            for pos in row_pos:
                set_pos.update(range(pos - 100, pos + 100))

            for pos in row_pos:
                gap = 112
                # positive
                begin = pos - 1 - gap
                end = pos - 1 + gap
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

                ins_position.append([begin, end])

            # delete
            delete_result_data = pd.read_csv(
                del_vcf_filename, sep="\t", index_col=0)
            delete_chromosome = delete_result_data[delete_result_data["CHROM"] == chromosome]
            row_pos = []
            row_end = []
            for index, row in delete_chromosome.iterrows():
                row_pos.append(row["POS"])
                row_end.append(row["END"])

            for pos in row_pos:
                set_pos.update(range(pos - 100, pos + 100))

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
        print("position end")

        # img/positive_cigar_img
        print("cigar start")
        if os.path.exists(data_dir + 'image/' + chromosome + '/negative_cigar_new_img' + '.pt') and not cigar_enforcement_refresh:
            print("loading")
            ins_cigar_img = torch.load(
                data_dir + 'image/' + chromosome + '/ins_cigar_new_img' + '.pt')
            del_cigar_img = torch.load(
                data_dir + 'image/' + chromosome + '/del_cigar_new_img' + '.pt')
            negative_cigar_img = torch.load(
                data_dir + 'image/' + chromosome + '/negative_cigar_new_img' + '.pt')
            # 由于未刷新数据增加的代码
            # all_p_img0 = positive_cigar_img[:, 0, :, :] + positive_cigar_img[:, 5, :, :]
            # all_n_img0 = negative_cigar_img[:, 0, :, :] + negative_cigar_img[:, 5, :, :]
            # positive_cigar_img = torch.cat([all_p_img0.unsqueeze(1), positive_cigar_img[:, 1:3, :, :]], dim = 1)
            # negative_cigar_img = torch.cat([all_n_img0.unsqueeze(1), negative_cigar_img[:, 1:3, :, :]], dim = 1)
            # save_path = data_dir + 'image/' + chromosome
            # torch.save(positive_cigar_img, save_path + '/positive_cigar_img' + '.pt')
            # torch.save(negative_cigar_img, save_path + '/negative_cigar_img' + '.pt')
            # end 从头跑程序需注释
        else:
            # sam_file = pysam.AlignmentFile(bam_path, "rb")
            ins_cigar_img = torch.empty(len(ins_position), 1, hight, hight)
            del_cigar_img = torch.empty(len(del_position), 1, hight, hight)
            negative_cigar_img = torch.empty(len(n_position), 1, hight, hight)
            for i, b_e in enumerate(ins_position):
                # f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
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
                        print("Exception cigar_img_single_optimal" + str(zoom))
                #     try:
                #         positive_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
                #     except Exception as e:
                #         print(e)
                #         print("Exception cigar_img_single_optimal_time2sapce")
                #         try:
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
                #         except Exception as e:
                #             print(e)
                #             print("Exception cigar_img_single_optimal_time3sapce")
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])

                print("===== finish(ins_cigar_img) " +
                      chromosome + " " + str(i))

            for i, b_e in enumerate(del_position):
                # f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
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
                        print("Exception cigar_img_single_optimal" + str(zoom))
                #     try:
                #         positive_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
                #     except Exception as e:
                #         print(e)
                #         print("Exception cigar_img_single_optimal_time2sapce")
                #         try:
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
                #         except Exception as e:
                #             print(e)
                #             print("Exception cigar_img_single_optimal_time3sapce")
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])

                print("===== finish(del_position) " + chromosome + " " + str(i))

            for i, b_e in enumerate(n_position):
                # f negative_cigar_img = torch.cat((negative_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
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
                        print("Exception cigar_img_single_optimal" + str(zoom))

                    # try:
                    #     negative_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
                    # except Exception as e:
                    #     print(e)
                    #     print("Exception cigar_img_single_optimal_time2sapce")
                    #     try:
                    #         negative_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
                    #     except Exception as e:
                    #         print(e)
                    #         print("Exception cigar_img_single_optimal_time3sapce")
                    #         negative_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])

                print("===== finish(n_position) " + chromosome + " " + str(i))
            # sam_file.close()

            save_path = data_dir + 'image/' + chromosome
            ut.mymkdir(save_path)
            torch.save(ins_cigar_img, save_path + '/ins_cigar_new_img' + '.pt')
            torch.save(del_cigar_img, save_path + '/del_cigar_new_img' + '.pt')
            torch.save(negative_cigar_img, save_path +
                       '/negative_cigar_new_img' + '.pt')

        print("cigar end")

        all_ins_cigar_img = torch.cat((all_ins_cigar_img, ins_cigar_img), 0)
        all_del_cigar_img = torch.cat((all_del_cigar_img, del_cigar_img), 0)
        all_negative_cigar_img = torch.cat(
            (all_negative_cigar_img, negative_cigar_img), 0)

    torch.save(all_ins_cigar_img, data_dir + '/all_ins_img' + '.pt')
    torch.save(all_del_cigar_img, data_dir + '/all_del_img' + '.pt')
    torch.save(all_negative_cigar_img, data_dir + '/all_n_img' + '.pt')

print("loading data")

all_ins_img = torch.load(data_dir + '/all_ins_img' + '.pt')
all_del_img = torch.load(data_dir + '/all_del_img' + '.pt')
all_n_img = torch.load(data_dir + '/all_n_img' + '.pt')

# all_ins_img = all_ins_cigar_img
# all_del_img = all_del_cigar_img
# all_n_img = all_negative_cigar_img

print("loaded")

length = len(all_ins_img) + len(all_del_img) + len(all_n_img)

ut.mymkdir(data_dir + "ins/")
ut.mymkdir(data_dir + "del/")
ut.mymkdir(data_dir + "n/")

for index in range(length):
    print(index)
    if index < len(all_ins_img):
        image = all_ins_img[index].clone()
        torch.save([image, 2], data_dir + "ins/" + str(index) + ".pt")
    elif index < len(all_ins_img) + len(all_del_img):
        index -= len(all_ins_img)
        image = all_del_img[index].clone()
        torch.save([image, 1], data_dir + "del/" + str(index) + ".pt")
    else:
        index -= len(all_ins_img) + len(all_del_img)
        image = all_n_img[index].clone()
        torch.save([image, 0], data_dir + "n/" + str(index) + ".pt")

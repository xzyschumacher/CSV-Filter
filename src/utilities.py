
import codecs
import csv
import os

import numpy as np
import torch
import torchvision
import pysam

from bed2image import trans2img
from torchvision import transforms

hight = 224
resize = torchvision.transforms.Resize([hight, hight])

def data_write_csv(file_name, datas): 
    file_csv = codecs.open(file_name, 'w+', 'utf-8') 
    writer = csv.writer(file_csv, delimiter=' ',
                        quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("file save successfule, processing")


def get_rms(records):
    return np.sqrt(sum([x ** 2 for x in records]) / len(records))


def get_cv(records): 
    mean = np.mean(records)
    std = np.std(records)
    cv = std / mean
    return mean, std, cv


def mid_list2img(mid_sign_list, chromosome): 
    mid_sign_img = torch.zeros(len(mid_sign_list), 9)
    for i, mid_sign in enumerate(mid_sign_list):
        if i % 50000 == 0:
            print(str(chromosome) + "\t" + str(i))
        mid_sign_img[i, 0] = len(mid_sign)
        if mid_sign_img[i, 0] == 1:
            continue
        mid_sign = np.array(mid_sign)
        mid_sign_img[i, 1], mid_sign_img[i, 2], mid_sign_img[i, 3], mid_sign_img[i,
                                                                                 4] = np.quantile(mid_sign, [0.25, 0.5, 0.75, 1], interpolation='linear')
        mid_sign_img[i, 5] = get_rms(mid_sign)
        mid_sign_img[i, 6], mid_sign_img[i,
                                         7], mid_sign_img[i, 8] = get_cv(mid_sign)

    return mid_sign_img


def mymkdir(mydir):
    if not os.path.exists(mydir): 
        os.makedirs(mydir)


def preprocess(bam_path, chromosome, chr_len, data_dir):

    return trans2img(bam_path, chromosome, chr_len, data_dir)


def to_input_image_single(img): 
    ims = torch.empty(len(img), hight, hight)

    for i, img_dim in enumerate(img):
        img_min = torch.min(img_dim)
        img_hight = torch.max(img_dim) - img_min + 2
        im = torch.zeros(len(img_dim), img_hight)

        for x in range(len(img_dim)):
            im[x, :img_dim[x] - img_min + 1] = img_hight

        ims[i] = resize(im.unsqueeze(0))

    return ims

class IdentifyDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):

        self.insfile_list = os.listdir(path + "ins")
        self.delfile_list = os.listdir(path + "del")
        self.nfile_list = os.listdir(path + "n")
        self.path = path
        self.transform = transform

        self._len = len(self.insfile_list) + \
            len(self.delfile_list) + len(self.nfile_list)

    def __len__(self):
        return self._len

    def __getitem__(self, index): #TODO
        if index < len(self.insfile_list):
            
            x, y = torch.load(self.path + "ins/" + self.insfile_list[index])
            x = self.transform(x)
            return x, y

        elif index < len(self.insfile_list) + len(self.delfile_list):
            index -= len(self.insfile_list)

            x, y = torch.load(self.path + "del/" + self.delfile_list[index])
            x = self.transform(x)
            return x, y

        else:
            index -= len(self.insfile_list) + len(self.delfile_list)

            x, y = torch.load(self.path + "n/" + self.nfile_list[index])
            x = self.transform(x)
            return x, y

def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x

def cigar_img_single_optimal(sam_file, chromosome, begin, end):
    read_length = []
    gap = "nan"

    for read in sam_file.fetch(chromosome, begin, end):

        if gap == "nan":
            gap = read.reference_start - begin

        read_list_terminal = 0
        empty = read.reference_start - begin
        if gap >= 0:
            read_list_terminal += empty
        else:
            read_list_terminal += empty - gap

        for operation, length in read.cigar: 
            if operation == 0 or operation == 1 or operation == 2 or operation == 4 or operation == 5 or operation == 8:
                read_list_terminal += length

        read_length.append(read_list_terminal)

    if read_length:
        mean = np.mean(read_length)
        std = np.std(read_length)
        maximum = int(mean + 3 * std) 

        cigars_img = torch.zeros([4, len(read_length), maximum])

        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            max_terminal = 0

            empty = read.reference_start - begin
            if gap >= 0:
                max_terminal = empty
            else:
                max_terminal = empty - gap

            for operation, length in read.cigar:
                if operation == 0:
                    if max_terminal+length < maximum:
                        cigars_img[0, i,
                                   max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[0, i, max_terminal:] = 255
                        break
                elif operation == 1:
                    if max_terminal+length < maximum:
                        cigars_img[1, i,
                                   max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[1, i, max_terminal:] = 255
                        break
                elif operation == 2:
                    if max_terminal+length < maximum:
                        cigars_img[2, i,
                                   max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[2, i, max_terminal:] = 255
                        break
                elif operation == 4:
                    if max_terminal+length < maximum:
                        cigars_img[3, i,
                                   max_terminal:max_terminal+length] = 255
                        max_terminal += length
                    else:
                        cigars_img[3, i, max_terminal:] = 255
                        break
                elif operation == 8:
                    if max_terminal+length < maximum:
                        max_terminal += length
                    else:
                        break
        cigars_img = resize(cigars_img)
    else:
        cigars_img = torch.zeros([4, hight, hight])

    return cigars_img


def kernel_cigar(read, ref_min, ref_max, cigar_resize, zoom):
    cigars_img = torch.zeros([1, int((ref_max - ref_min) / zoom)])

    max_terminal = read.reference_start - ref_min

    tau = 1

    for operation, length in read.cigar:
        if operation == 0:
            cigars_img[0, int(max_terminal / zoom):int((max_terminal+length) / zoom)] = 63/tau
            max_terminal += length
        elif operation == 2: 
            cigars_img[0, int(max_terminal / zoom):int((max_terminal+length) / zoom)] = 127/tau
            max_terminal += length
        elif operation == 1: 
            cigars_img[0, int((max_terminal - length / 2) / zoom):int((max_terminal + length / 2) / zoom)] = 191/tau
        elif operation == 4: 
            cigars_img[0, int((max_terminal - length / 2) / zoom):int((max_terminal + length / 2) / zoom)] = 255/tau

        elif operation == 3 or operation == 7 or operation == 8:
            max_terminal += length

    return cigar_resize(cigars_img.unsqueeze(1))

def cigar_new_img_single_optimal(bam_path, chromosome, begin, end, zoom):
    r_start = []
    r_end = []
    sam_file = pysam.AlignmentFile(bam_path, "rb")

    for read in sam_file.fetch(chromosome, begin, end):
        if read.reference_start is not None:
            r_start.append(read.reference_start)

        if read.reference_end is not None:
            r_end.append(read.reference_end)

    if r_start:
        ref_min = np.min(r_start)
        ref_max = np.max(r_end)
        cigars_img = torch.empty([1, len(r_start), hight])
        cigar_resize = torchvision.transforms.Resize([1, hight])

        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            cigars_img[:, i:i + 1,
                       :] = kernel_cigar(read, ref_min, ref_max, cigar_resize, zoom)

        cigars_img = resize(cigars_img)
    else:
        cigars_img = torch.zeros([1, hight, hight])

    sam_file.close()
    
    return cigars_img

def cigar_new_img_single_memory(bam_path, chromosome, begin, end):
    r_start = []
    r_end = []
    sam_file = pysam.AlignmentFile(bam_path, "rb")

    for read in sam_file.fetch(chromosome, begin, end):
        r_start.append(read.reference_start)
        r_end.append(read.reference_end)

    if r_start:
        ref_min = np.min(r_start)
        ref_max = np.max(r_end)

        cigars_img = torch.zeros([1, len(r_start), ref_max - ref_min])
        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            max_terminal = read.reference_start - ref_min

            for operation, length in read.cigar:
                if operation == 0:
                    cigars_img[0, i, max_terminal:max_terminal+length] = 255
                    max_terminal += length
                elif operation == 2:
                    max_terminal += length
                elif operation == 3 or operation == 7 or operation == 8:
                    max_terminal += length

        cigars_img1 = resize(cigars_img)

        cigars_img[:, :, :] = 0
        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            max_terminal = read.reference_start - ref_min

            for operation, length in read.cigar:
                if operation == 0:
                    max_terminal += length
                elif operation == 2:
                    cigars_img[0, i, max_terminal:max_terminal+length] = 255
                    max_terminal += length

                elif operation == 3 or operation == 7 or operation == 8:
                    max_terminal += length
        cigars_img2 = resize(cigars_img)

        cigars_img[:, :, :] = 0
        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            max_terminal = read.reference_start - ref_min

            for operation, length in read.cigar:
                if operation == 0:
                    max_terminal += length
                elif operation == 2:
                    max_terminal += length
                elif operation == 1:
                    cigars_img[0, i, max_terminal -
                               int(length / 2):max_terminal + int(length / 2)] = 255

                elif operation == 3 or operation == 7 or operation == 8:
                    max_terminal += length
        cigars_img3 = resize(cigars_img)

        cigars_img[:, :, :] = 0
        for i, read in enumerate(sam_file.fetch(chromosome, begin, end)):
            max_terminal = read.reference_start - ref_min

            for operation, length in read.cigar:
                if operation == 0:
                    max_terminal += length
                elif operation == 2:
                    max_terminal += length
                elif operation == 4:
                    cigars_img[0, i, max_terminal -
                               int(length / 2):max_terminal + int(length / 2)] = 255

                elif operation == 3 or operation == 7 or operation == 8:
                    max_terminal += length
        cigars_img4 = resize(cigars_img)

        cigars_img = torch.empty([4, hight, hight])
        cigars_img[0] = cigars_img1
        cigars_img[1] = cigars_img2
        cigars_img[2] = cigars_img3
        cigars_img[3] = cigars_img4

    else:
        cigars_img = torch.zeros([4, hight, hight])

    sam_file.close()
    return cigars_img

def to_img_mid_single(img, hight=224):

    img = torch.maximum(img.float(), torch.tensor(0))

    pic_length = img.size()
    im = torch.zeros(4, pic_length[-1], hight)

    for x in range(pic_length[-1]):
        y = img[:, x].int()
        for j in range(pic_length[0]):
            im[j, x, :y[j]] = 255
            y_float = img[j, x] - y[j]
            im[j, x, y[j]:y[j]+1] = torch.round(255 * y_float)

    im = resize(im)

    return im

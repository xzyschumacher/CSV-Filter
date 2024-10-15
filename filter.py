from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from scipy import interp
from tqdm import tqdm

import pandas as pd
import torch
from net import IDENet
import sys
import os


if len(sys.argv) < 2:
    print("usage error!")
    exit()

seed_everything(2022)

root_dir = "../"
vcf_data_dir = "../data/"
vcf_name = "sniffles2-CLR-minimap2"

config = {
    "lr": 7.1873e-06,
    "batch_size": 64, 
    "beta1": 0.9,
    "beta2": 0.999,
    'weight_decay': 0.0011615,
    'model': sys.argv[1]
}

model = IDENet.load_from_checkpoint(
    root_dir + "models/" + config['model'] + ".ckpt", path=root_dir + "data/", config=config)

trainer = pl.Trainer(gpus=1)

model.eval()
result = trainer.test(model)

output = torch.load("result.pt")

y = torch.empty(0, 3)
y_hat = torch.empty(0, 3).cuda()


for out in output:
    for ii in out['y']:
        if ii == 0:
            y = torch.cat([y, torch.tensor([1, 0, 0]).unsqueeze(0)], 0)
        elif ii == 1:
            y = torch.cat([y, torch.tensor([0, 1, 0]).unsqueeze(0)], 0)
        else:
            y = torch.cat([y, torch.tensor([0, 0, 1]).unsqueeze(0)], 0)
    y_hat = torch.cat([y_hat, out['y_hat']], 0)

y_test = y.cpu().numpy()
y_score = y_hat.cpu().numpy()
n_classes = y.shape[1]

del_df = pd.read_csv(vcf_data_dir + vcf_name + ".vcf_del", sep="\t")
ins_df = pd.read_csv(vcf_data_dir + vcf_name + ".vcf_ins", sep="\t")

filtered_del_df = pd.DataFrame(columns=del_df.columns)
filtered_ins_df = pd.DataFrame(columns=ins_df.columns)

predicted_labels = torch.argmax(y_hat, dim=1)

for i in range(len(del_df)):
    if i < len(predicted_labels):
        predicted_label = predicted_labels[i].item()
        if predicted_label == 0:
            filtered_del_df = filtered_del_df.append(del_df.iloc[i])
        elif predicted_label == 1:
            filtered_del_df = filtered_del_df.append(del_df.iloc[i])
        elif predicted_label == 2:
            temp_row = del_df.iloc[i].copy()
            temp_row['ALT'] = '<INS>'
            temp_row['INFO'] = temp_row['INFO'].replace('SVTYPE=DEL', 'SVTYPE=INS')
            filtered_del_df = filtered_del_df.append(temp_row)
            continue

for i in range(len(ins_df)):
    if i < len(predicted_labels):
        predicted_label = predicted_labels[i].item()
        if predicted_label == 0:
            filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])
        elif predicted_label == 1:
            temp_row = ins_df.iloc[i].copy()
            temp_row['ALT'] = '<DEL>'
            temp_row['INFO'] = temp_row['INFO'].replace('SVTYPE=INS', 'SVTYPE=DEL')
            filtered_ins_df = filtered_ins_df.append(temp_row)
            continue
        elif predicted_label == 2:
            filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])

filtered_del_df.to_csv(vcf_data_dir + vcf_name + "_filtered.vcf_del", sep="\t", index=False)
filtered_ins_df.to_csv(vcf_data_dir + vcf_name + "_filtered.vcf_ins", sep="\t", index=False)

with open(vcf_data_dir + vcf_name + ".vcf") as f:
    lines = f.readlines()

header_lines = [line for line in lines if line.startswith("#")]
data_lines = [line for line in lines if not line.startswith("#")]

with open(root_dir + "data/data_temp.vcf", "w") as f:
    for line in data_lines:
        f.write(line)

vcf_df = pd.read_csv(root_dir + "data/data_temp.vcf", sep="\t", header=None)
filtered_del_df = pd.read_csv(vcf_data_dir + vcf_name + "_filtered.vcf_del", sep="\t", skiprows=1, header=None)

ins_file_path = vcf_data_dir + vcf_name + "_filtered.vcf_ins"

try:
    filtered_ins_df = pd.read_csv(ins_file_path, sep="\t", skiprows=1, header=None)
except pd.errors.EmptyDataError:
    filtered_ins_df = pd.DataFrame()

filtered_df = pd.concat([filtered_del_df, filtered_ins_df])

vcf_df['index'] = vcf_df[0].astype(str) + "_" + vcf_df[1].astype(str)
filtered_df['index'] = filtered_df[0].astype(str) + "_" + filtered_df[1].astype(str)

indices_to_keep = set(filtered_df['index'].values)
vcf_df_filtered = pd.DataFrame()

for i in tqdm(range(len(vcf_df))):
    if vcf_df.iloc[i]['index'] in indices_to_keep:
        vcf_df_filtered = vcf_df_filtered.append(vcf_df.iloc[i])

import os
os.remove(root_dir + "data/data_temp.vcf")

vcf_df_filtered = vcf_df_filtered.drop(columns=['index'])

vcf_df_filtered[1] = vcf_df_filtered[1].astype(int)

with open(root_dir + "/data/" + vcf_name + "_INS_DEL_filtered.vcf", "w") as f:
    for line in header_lines:
        f.write(line)
    vcf_df_filtered.to_csv(f, sep="\t", header=False, index=False)

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


if len(sys.argv) < 2:
    print("usage error!")
    exit()

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
seed_everything(2022)

# data_dir = "../datasets/NA12878_PacBio_MtSinai/"
root_dir = "../"

config = {
    "lr": 7.1873e-06,
    "batch_size": 118,  # 14,
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

# 读取del和ins文件
del_df = pd.read_csv(root_dir + "data/output_sniffles.vcf_del", sep="\t")
ins_df = pd.read_csv(root_dir + "data/output_sniffles.vcf_ins", sep="\t")

# 初始化新的DataFrame来保存过滤后的结果
filtered_del_df = pd.DataFrame(columns=del_df.columns)
filtered_ins_df = pd.DataFrame(columns=ins_df.columns)

# 获取分类结果
predicted_labels = torch.argmax(y_hat, dim=1)

# 对del文件进行操作
for i in range(len(del_df)):
    predicted_label = predicted_labels[i].item()   # 取出张量的值
    # if predicted_label == 0:
    #      filtered_del_df = filtered_del_df.append(del_df.iloc[i])
    # elif predicted_label == 1:
    #      filtered_del_df = filtered_del_df.append(del_df.iloc[i])
    # else:
    #     continue

    if predicted_label == 0:
        filtered_del_df = filtered_del_df.append(del_df.iloc[i])
        # continue
    elif predicted_label == 1:
        filtered_del_df = filtered_del_df.append(del_df.iloc[i])
    elif predicted_label == 2:
        # temp_row = del_df.iloc[i].copy()
        # temp_row['ALT'] = '<INS>'
        # temp_row['INFO'] = temp_row['INFO'].replace('SVTYPE=DEL', 'SVTYPE=INS')
        # filtered_del_df = filtered_del_df.append(temp_row)
        continue

# 对ins文件进行操作
for i in range(len(ins_df)):
    predicted_label = predicted_labels[i].item()   # 取出张量的值
    # if predicted_label == 0:
    #      filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])
    # elif predicted_label == 2:
    #      filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])
    # else:
    #     continue

    if predicted_label == 0:
        filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])
        # continue
    elif predicted_label == 1:
        # temp_row = ins_df.iloc[i].copy()
        # temp_row['ALT'] = '<DEL>'
        # temp_row['INFO'] = temp_row['INFO'].replace('SVTYPE=INS', 'SVTYPE=DEL')
        # filtered_ins_df = filtered_ins_df.append(temp_row)
        continue
    elif predicted_label == 2:
        filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])

# # 保存结果
filtered_del_df.to_csv(root_dir + "data/output_sniffles_filtered.vcf_del", sep="\t", index=False)
filtered_ins_df.to_csv(root_dir + "data/output_sniffles_filtered.vcf_ins", sep="\t", index=False)

# # 修改原始vcf文件
# # 读取元数据和列标题行
# with open(root_dir + "data/output_sniffles.vcf", "r") as f:
#     lines = [l for l in f if l.startswith("##")]
# #     col_line = [l for l in f if l.startswith("#")]

# # # 如果没找到列标题行，输出错误信息
# # if len(col_line) == 0:
# #     print("Error: Column title line starting with # not found in the VCF file.")
# #     exit(1)

# # 读取数据行
# vcf_df = pd.read_csv(root_dir + "data/output_sniffles.vcf", sep="\t", comment="#", header=None)
# filtered_del_df = pd.read_csv(root_dir + "data/output_sniffles_filtered.vcf_del", sep="\t", skiprows=1, header=None)
# filtered_ins_df = pd.read_csv(root_dir + "data/output_sniffles_filtered.vcf_ins", sep="\t", skiprows=1, header=None)

# # 合并del和ins的结果
# filtered_df = pd.concat([filtered_del_df, filtered_ins_df])

# # 用filtered_df中的数据替换vcf_df中的数据
# for i in tqdm(range(len(vcf_df))):
#     vcf_row = vcf_df.iloc[i]
#     filtered_row = filtered_df[(filtered_df[0] == vcf_row[0]) & (filtered_df[1] == vcf_row[1])]

#     if len(filtered_row) == 0:
#         continue

#     filtered_row = filtered_row.iloc[0]

#     if vcf_row[4] != filtered_row[4] or "SVTYPE=" + vcf_row[7].split("SVTYPE=")[1].split(";")[0] != "SVTYPE=" + filtered_row[7].split("SVTYPE=")[1].split(";")[0]:
#         vcf_df.iloc[i] = filtered_row

# # 保存结果
# with open(root_dir + "data/output_sniffles_filtered.vcf", "w") as f:
#     for line in lines:
#         f.write(line)
#     #f.write(col_line[0])
#     vcf_df.to_csv(f, sep="\t", header=False, index=False)

# 读取数据行并保留头部
with open(root_dir + "data/output_sniffles.vcf") as f:
    lines = f.readlines()

header_lines = [line for line in lines if line.startswith("#")]
data_lines = [line for line in lines if not line.startswith("#")]

with open(root_dir + "data/data_temp.vcf", "w") as f:
    for line in data_lines:
        f.write(line)

# 读取处理后的数据
vcf_df = pd.read_csv(root_dir + "data/data_temp.vcf", sep="\t", header=None)
filtered_del_df = pd.read_csv(root_dir + "data/output_sniffles_filtered.vcf_del", sep="\t", skiprows=1, header=None)
filtered_ins_df = pd.read_csv(root_dir + "data/output_sniffles_filtered.vcf_ins", sep="\t", skiprows=1, header=None)

# 合并del和ins的结果
filtered_df = pd.concat([filtered_del_df, filtered_ins_df])

# 创建索引列
vcf_df['index'] = vcf_df[0].astype(str) + "_" + vcf_df[1].astype(str)
filtered_df['index'] = filtered_df[0].astype(str) + "_" + filtered_df[1].astype(str)

# 根据索引过滤vcf_df
indices_to_keep = set(filtered_df['index'].values)
vcf_df_filtered = pd.DataFrame()

for i in tqdm(range(len(vcf_df))):
    if vcf_df.iloc[i]['index'] in indices_to_keep:
        vcf_df_filtered = vcf_df_filtered.append(vcf_df.iloc[i])

# 删除临时文件
import os
os.remove(root_dir + "data/data_temp.vcf")

# 删除索引列
vcf_df_filtered = vcf_df_filtered.drop(columns=['index'])

# Convert POS column back to int
vcf_df_filtered[1] = vcf_df_filtered[1].astype(int)

# 保存结果
with open(root_dir + "data/output_sniffles_filtered.vcf", "w") as f:
    for line in header_lines:
        f.write(line)
    vcf_df_filtered.to_csv(f, sep="\t", header=False, index=False)

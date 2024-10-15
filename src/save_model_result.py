from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from scipy import interp
from sklearn.metrics import auc, classification_report, roc_curve

import torch
from net import IDENet
import sys

import pickle


if len(sys.argv) < 2:
    print("usage error!")
    exit()

seed_everything(2022)

root_dir = "../"

config = {
    "lr": 7.1873e-06,
    "batch_size": 118, 
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

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

with open('model_result-resnet200x2-self.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)

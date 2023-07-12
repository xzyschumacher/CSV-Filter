from itertools import cycle

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from net import IDENet
import sys

root_dir = "../"

data = open("result.pt.txt",'w',encoding="utf-8")

output = torch.load("result.pt")

print(output,file=data)

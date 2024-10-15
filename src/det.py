import torch
import torchvision
from pudb import set_trace

model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 2, pretrained_backbone=True, trainable_backbone_layers = 5)
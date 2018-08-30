import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from utils import data
from sklearn.metrics import roc_auc_score


# checkpoint directory
chkpt_path = 'model.pth.tar'
num_classes = 14
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]
data_dir = "./data/chexnet-data/images"
test_image_list = "./data/chexnet-data/test_list.txt"
batch_size = 64

def main():
    
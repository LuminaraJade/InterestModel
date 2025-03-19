import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from model.DeepFM import DeepFM
from dataset_self import DeepFMDataset

# 900000 items for training, 10000 items for valid, of all 1000000 items
Num_train = 9000


# load data
train_data = DeepFMDataset('./data', train=True)
loader_train = DataLoader(train_data, batch_size=16,
                          sampler=sampler.SubsetRandomSampler(range(Num_train)))
val_data = DeepFMDataset('./data', train=True)
loader_val = DataLoader(val_data, batch_size=16,
                        sampler=sampler.SubsetRandomSampler(range(Num_train, 10000)))


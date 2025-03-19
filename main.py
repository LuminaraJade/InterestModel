import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset

# 900000 items for training, 10000 items for valid, of all 1000000 items
Num_train = 2300000

# load data
train_data = CriteoDataset('/media/admin/9ee79a49-cc36-4fd2-8ca4-506d17c6e480/data/yangziqi/interest_model/DeepFM2/data', train=True)
loader_train = DataLoader(train_data, batch_size=10000,
                          sampler=sampler.SubsetRandomSampler(range(Num_train)))
# loader_train = DataLoader(train_data, batch_size=10000)
val_data = CriteoDataset('/media/admin/9ee79a49-cc36-4fd2-8ca4-506d17c6e480/data/yangziqi/interest_model/DeepFM2/data', train=True)
loader_val = DataLoader(val_data, batch_size=10000,
                        sampler=sampler.SubsetRandomSampler(range(Num_train, 2687732)))
# loader_val = DataLoader(val_data, batch_size=10000)


feature_sizes = pd.read_pickle('/media/admin/9ee79a49-cc36-4fd2-8ca4-506d17c6e480/data/yangziqi/interest_model/DeepFM2/feature_size.pkl').values
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

model = DeepFM(feature_sizes, use_cuda=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
model.fit(loader_train, loader_val, optimizer, epochs=2000, verbose=True)
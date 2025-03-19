import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

continous_features = 13

class DeepFMDataset(Dataset):
    def __init__(self,filepath,train = True):

        self.train = train

        if self.train == True:
            data = pd.read_csv(os.path.join(filepath,'train.txt'))
            self.train_data = data.iloc[:,:-1].values
            self.target = data.iloc[:,-1].values
        else:
            data = pd.read_csv(os.path.join(filepath,'test.txt'))
            self.test_data = data.iloc[:,:-1].values

    
    def __getitem__(self, index):
        if self.train == True:
            feature_i, target_i = self.train_data[index, :], self.target[index]
            # 连续特征的索引部分设为全0数组
            Xi_coutinous = np.zeros_like(feature_i[:continous_features])
            # 类别特征的索引部分保留原始值
            Xi_categorial = feature_i[continous_features:]
            Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial.astype(np.int32)))) # Xi表示特征的索引

            # 连续特征的值直接保留
            Xv_coutinous = feature_i[:continous_features]
            # 类别特征的值设为全1数组，表示每个类别特征的值为1（one-hot编码）
            Xv_categorial = np.ones_like(feature_i[continous_features:])
            Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32)).unsqueeze(-1)
            return Xi, Xv, target_i
    
        else:
            feature_i = self.test_data.iloc[index, :]
            Xi_coutinous = np.ones_like(feature_i[:continous_features])
            Xi_categorial = feature_i[continous_features:]
            Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)
            
            Xv_categorial = np.ones_like(feature_i[continous_features:])
            Xv_coutinous = feature_i[:continous_features]
            Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
            return Xi, Xv

  
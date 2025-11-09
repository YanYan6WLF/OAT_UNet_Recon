# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       ReconDataset.py
   Project Name:    project_name
   Author :         Yan
   Date:            2025/11/08
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/11/08:
-------------------------------------------------
"""

import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import scipy.io as scio 

class ReconDataset(Dataset):
    def __init__(self, dataset_pathr, n_iter,transforms_=None, select=True,iter=None): # for iter in range n_iter
        if select:
            mode ='train'
        else:
            mode = 'test'

        self.files = sorted(glob.glob(os.path.join(dataset_pathr, mode) + '/*.mat')) # load data from train set and test set

        if transforms_:
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = None

        self.n_iter=n_iter 
        self.iter =iter

    def __getitem__(self, index):

        mat_data = scio.loadmat(self.files[index % len(self.files)])

        reconBP = mat_data['recon_het2'] 
        FrangiFil=mat_data['recon_BPhet2_Frangi_filter']
        reconMB_all = mat_data['recon_MBCPU_het2_lsqr_all']
        A=mat_data['A_mat']
        b=mat_data['sigMat_fil_MB']

        rawdata = torch.zeros((1, 125, 125)) 
        # The DataLoader will add a batch dimension, eventually becoming (batch, 1, 125, 125), causing the model to forward with an error.
        lsqr_data=reconMB_all[:,:,:self.n_iter]
        lsqr_data = torch.from_numpy(lsqr_data).float().unsqueeze(0)


        if self.transform:
            rawdata = self.transform(rawdata)  
            
        return rawdata,lsqr_data,A,b 

    def __len__(self):
        return len(self.files)

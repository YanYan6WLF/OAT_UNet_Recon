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
    def __init__(self, dataset_pathr, transforms_=None, select=True,n_iter=11,iter=None): # for iter in range n_iter
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
        A=mat_data['A']
        b=mat_data['sigMat_mean']


        if self.iter==0:
            rawdata=b
        else:
            rawdata=reconMB_all[:,:,self.iter]

        lsqr_data=reconMB_all[:,:,self.iter+1]

        if not isinstance(rawdata, torch.Tensor):
            rawdata = torch.from_numpy(rawdata).float().unsqueeze(0)
            lsqr_data= torch.from_numpy(lsqr_data).float().unsqueeze(0)
            FrangiFil= torch.from_numpy(FrangiFil).float().unsqueeze(0)


        if self.transform:
            rawdata = self.transform(rawdata)  
            
        return rawdata,lsqr_data,A,b 

    def __len__(self):
        return len(self.files)

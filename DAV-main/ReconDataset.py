# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       ReconDataset.py
   Project Name:    Yan`s Project
   Author :         Yan
   Date:            2025/10/25
-------------------------------------------------
   Change Activity:
                   2025/10/25:
-------------------------------------------------
"""

import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import transforms
# 如果将来要用图像而不是 `.npy` 数组，可以考虑同时导入 `PIL.Image`。
import scipy.io as scio 

class ReconDataset(Dataset):
    def __init__(self, dataset_pathr, transforms_=None, select=True,n_iter=11,iter=None): # for iter in range n_iter
        if select:
            mode ='train'
        else:
            mode = 'test'
        self.files = sorted(glob.glob(os.path.join(dataset_pathr, mode) + '/*.mat'))

        if transforms_:
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = None

        self.n_iter=n_iter # eg, 11
        self.iter =iter

    def __getitem__(self, index):

        mat_data = scio.loadmat(self.files[index % len(self.files)])
         # numpy 只是一个 中间格式，因为：
            # 很多文件（如 .npy、.npz、.mat）只能用 numpy 读取；
            # 但是 PyTorch 的计算（模型前向传播、loss、反向传播）都要用 Tensor 类型。

        # 假设 mat 文件中有一个名为 'img' 的变量
        reconBP = mat_data['recon_het2'] #### prepare to be changed
        FrangiFil=mat_data['recon_BPhet2_Frangi_filter']
        reconMB_all = mat_data['recon_MBCPU_het2_lsqr_all']
        A=mat_data['A']
        b=mat_data['sigMat_mean']


        if self.iter==0:
            rawdata=b
        else:
            rawdata=reconMB_all[:,:,self.iter]

        out_data=reconMB_all[:,:,self.iter+1]

        # save('sample.mat', 'image', 'label')
        # mat = scio.loadmat(file_path)
        # img = mat['image']
        # label = mat['label']
        # return img, label

        # 转成 Tensor（除非 transform 已经做了）
        # 若你打算后续用 DataLoader 批量取数据，最好返回一个 torch.Tensor 而不是 numpy.ndarray
        if not isinstance(rawdata, torch.Tensor):
            rawdata = torch.from_numpy(rawdata).float().unsqueeze(0)
            out_data= torch.from_numpy(out_data).float().unsqueeze(0)
            

        if self.transform:
            rawdata = self.transform(rawdata)  # 应用数据变换
            
        return rawdata,out_data,A,b 

    def __len__(self):
        return len(self.files)

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       basicblock.py
   Project Name:    Basic Block
   Author :         Yan
   Date:            2025/12/01
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/12/01:
-------------------------------------------------
"""

import torch
import numpy as np
import torch.nn as nn

class BasicBlock1(nn.Module):
    def __init__(self):
        super(BasicBlock1,self).__init__()
        self.lambda_step=nn.Parameter(torch.tensor([0.1]))

    def forward(self,x, phi,b):
        '''
        x: (batch_size, channel=1, img_size,img_size)
        '''
        batch_size=x.size(0)  # x.shape[0]
        img_size=x.size(2)
        n_pixels=img_size*img_size
        # x=x.view(batch_size,n_pixels,-1) # batch matrix multiplication 这个其实也行的，每个batch单独做矩阵乘法
        x=x.view(batch_size, -1).t()  # (batch_size, n_pixels) flatten
        '''
            # 因为FISTA做矩阵乘法时
                # Phi @ x = b
                # (30, 4096) @ (4096, 32) = (30, 32)
                #              ↑ x的形状必须是(4096, batch_size)
        '''
        # phi : (batch_size, num_projections,n_pixels)
        # phiT=phi.permute(0,2,1)
        # phiTpi=phiT@ phi # (batch_size, n_pixels, num_projections) @ (batch_size, num_projections,n_pixels) = (batch_size, n_pixels, n_pixels)
        # b: (batchsize, n_pixels,1)
        # phTb= phiT@b # (batch_size, n_pixels, num_projections) @ (batch_size, num_projections,1) = (batch_size, n_pixels,1)
        phiTphi= phi.t() @ phi
        phiTb= phi.t() @ b

        x_new=x-self.lambda_step*(phiTphi@x-phiTb) # self.lambda_step is not a function, so no need to call it; not callable

        x_new=x_new.t().view(batch_size,1, img_size,img_size) # resize

        return x_new
    

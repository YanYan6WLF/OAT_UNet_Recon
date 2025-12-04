# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       FISTA_NET.py
   Project Name:    FISTA-NET
   Author :         Yan
   Date:            2025/12/01
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/12/01:
-------------------------------------------------
"""

import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt

def initial_weights(self): # 你的初始化函数写在 一个网络的 class 里面。 比如 ISTA_Net 或 BasicBlock：self = 当前的 ISTA_Net 这个模型
    for m in self.modules():
        if isinstance(m,nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias,0) #?
        elif isinstance(m,nn.BatchNorm2d):
            init.constant_(m.weight,1)
            init.constant_(m.bias,0)
        elif isinstance(m,nn.Linear):
            init.normal_(m.weight,0,0.001) # weight 从均值 0、标准差 0.01 的正态分布取值→ 初始化非常小、非常稳定
            init.constant_(m.bias,0)


class BasicBlock(nn.Module):
    def __init__(self,features=8):  # To be change
        super(BasicBlock,self).__init__()
        self.sp=nn.Softplus()

        # To be changed
        self.conv_D=nn.Conv2d(1, features, kernel_size=3, padding=1)
        self.conv1_forward=nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2_forward=nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv3_forward=nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv4_forward=nn.Conv2d(features, features, kernel_size=3, padding=1)

        self.conv1_backward=nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2_backward=nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv3_backward=nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv4_backward=nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_G=nn.Conv2d(features, 1, kernel_size=3, padding=1)

    def forward(self,lambda_step, theta,x, phiTpi, phiTb): # input: x: [batch_size,1,img_size,img_size]


        # ---------------------- size ----------------------
        img_size=x.shape[2]
        batch_size=x.shape[0]
        n_pixels=img_size*img_size
        
        x=x.view(batch_size, 1,n_pixels,-1) # (batch_size,1, n_pixels,1)
        x=torch.squeeze(x,1)  # (batch_size, n_pixels,1)
        # x = torch.squeeze(x, 2).t()             
        # x = mask.mm(x)  ??

        #----------------------- Gradient descent step ----------------------
        x_new = x - lambda_step*(phiTpi @ x - phiTb)  # phiTpi: (batch_size, n_pixels, n_pixels); phiTb: (batch_size, n_pixels,1)

        # quadratic tv gradient descent from doi:  10.1109/TMI.2009.2022540 Eq. (10)
        # x = x - self.Sp(lambda_step) * torch.inverse(PhiTPhi + 0.001 * LTL).mm(PhiTPhi.mm(x) - PhiTb - 0.001 * LTL.mm(x))


        # ------------------------ Reshape ----------------------
        x_new=x_new.view(batch_size,1, img_size,img_size)  # (batch_size,1,img_size,img_size)
        # x = torch.mm(mask.t(), x)
        # x = x.view(pnum, pnum, -1)
        # x = x.unsqueeze(0)
        # x_input = x.permute(3, 0, 1, 2)

        # ---------------------- Denoising step / CNN Regularization ----------------------
        # To be changed
        x_D=self.conv_D(x_new)
        x=self.conv1_forward(x_D)
        x= F.relu(x)
        x=self.conv2_forward(x)
        x= F.relu(x)
        x=self.conv3_forward(x)
        x= F.relu(x)
        x_forward=self.conv4_forward(x)

        # softthresholding
        x_st=torch.mul(torch.sign(x_forward),F.relu(torch.abs(x_forward)-theta))

        x=self.conv1_backward(x_st)
        x= F.relu(x)
        x=self.conv2_backward(x)
        x= F.relu(x)
        x=self.conv3_backward(x)
        x= F.relu(x)
        x_backward=self.conv4_backward(x)
        x_G=self.conv_G(x_backward)

        #  (skip connection); non-negative output
        x_pred=F.relu(x_G+x_new)


        # ---------------------- Symmetry Loss Components ----------------------

        x=self.conv1_backward(x_forward)
        x= F.relu(x)
        x=self.conv2_backward(x)
        x= F.relu(x)
        x=self.conv3_backward(x)
        x= F.relu(x)
        x_D_est=self.conv4_backward(x)
        Lsym=x_D_est-x_D # Symmetric loss 本来就应该在“feature space（特征空间）”计算，而不是在图像空间计算。 Symmetric loss 不是检查图像是否一样，而是检查 CNN 的 forward path 和 backward path 是否对称。
        # 为了让 CNN 更稳定，论文里说： 希望 backward network 反推回来的特征 x_D_est 能够接近 forward network 得到的特征 x_D
        # 如果 forward 和 backward “形状一样但数值完全乱飞”，模型会不稳定

        return [x_pred, Lsym, x_st] # x_pred: [batch_size, 1, img_Size, img_Size]


# ---------------------------------------------------FISTA-NET---------------------------------------------------

class FISTANet(nn.Module):
    def __init__(self, num_layers):
        super(FISTANet,self).__init__()
        self.num_layers=num_layers
        # self.L = L
        # self.mask =mask

        self.basicblock=BasicBlock(features=32)  # To be changed
        self.layers=nn.ModuleList( [self.basicblock for _ in range(num_layers)] )
        self.layers.apply(initial_weights)
        # 参数少 → 泛化能力强
        # 适合医学图像重建（数据通常不多）

        # ----------------------- Parameters to be Learned ----------------------
        self.sp=nn.Softplus()

        self.w_mu=nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu=nn.Parameter(torch.Tensor([0.1]))

        self.w_theta=nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta=nn.Parameter(torch.Tensor([-2.0]))

        self.w_rho=nn.Parameter(torch.Tensor([0.5]))
        self.b_rho=nn.Parameter(torch.Tensor([0]))


    def forward(self, phi, b, x0):
    
        # phi: (batch_size,1, num_projections,n_pixels)
        # b: (batch_size, 1,num_projections,1)
        phi=torch.squeeze(phi,1) # (batch_size, num_projections,n_pixels)
        b=torch.squeeze(b,1)     # (batch_size, num_projections,1)
        phiT=phi.permute(0,2,1)
        phiTpi=phiT@ phi # (batch_size, n_pixels, num_projections) @ (batch_size, num_projections,n_pixels) = (batch_size, n_pixels, n_pixels)
        phiTb= phiT@b # (batch_size, n_pixels, num_projections) @ (batch_size, num_projections,1) = (batch_size, n_pixels,1)

        # b = torch.squeeze(b, 1)
        # b = torch.squeeze(b, 2)
        # b = b.t()

        # PhiTPhi = self.Phi.t().mm(self.Phi)
        # PhiTb = self.Phi.t().mm(b)
        # LTL = self.L.t().mm(self.L)
        xold=x0
        y=xold

        layers_sym=[] # Save symmetry loss for each layer
        layers_st=[]  # Save soft-thresholded x for each layer  # for computing sparsity constraint
        xnews=[] # list 

        xnews.append(x0)

        for  i , layer in enumerate(self.layers):
            
            # ----------------------- Learnable Parameters ----------------------
            # lambda_step
            mu=self.sp(self.w_mu*i + self.b_mu)
            # threshold theta
            theta=self.sp(self.w_theta*i + self.b_theta)

            # ----------------------- FISTA-NET Layer Forward ----------------------
            [xnew, Lsym, x_st]= layer(mu, theta, y, phiTpi, phiTb)  # xnew: (batch_size,1,img_size,img_size)
            # rho for momentum update
            layers_sym.append(Lsym)
            layers_st.append(x_st)
            xnews.append(xnew)

            rho=(self.sp(self.w_rho*i + self.b_rho)-self.sp(self.w_rho+self.b_rho))/self.sp(self.w_rho*i+self.b_rho)  # normalize rho to [0,1)
            y=xnew+rho*(xnew - xold)
            xold=xnew

        return [xnews, layers_sym, layers_st]
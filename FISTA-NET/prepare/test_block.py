# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       test_block.py
   Project Name:    test block
   Author :         Yan
   Date:            2025/12/01
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/12/01:
-------------------------------------------------
"""

import numpy as np
import torch 
import matplotlib.pyplot as plt
from basicblock3 import BasicBlock3

def test_basic_block(block, phi ,b , num_iters=5): ##  parameter without a default follows parameter with a default; 有默认值的参数必须放在最后！

    img_size=64
    # x0=torch.zeros(batch_size,1,img_size,img_size_
    x=torch.zeros(1,1,img_size,img_size) # initial guess x not x0

    # batch_size 
    phiTphi= phi.t() @ phi
    phiTb= phi.t() @ b

    results=[x.detach().numpy().squeeze()]
    
    for i in range(num_iters):
        x = block(x,phiTphi=phiTphi,phiTb=phiTb)
        results.append(x.detach().numpy().squeeze())

        x_flat=x.view(-1,1)
        loss=torch.norm(phi@x_flat - b).item() # 不能detach,需要保留梯度，才能反向传播
        print(f"iteration:{i}, loss: {loss:.4f}")


    fig,axes=plt.subplots(1,len(results), figsize=(15,5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(results[i], cmap='gray')
        ax.set_title(f"Iter {i}")
    plt.tight_layout() # ! adjust spacing
    plt.show()

    return results


# 是的，CNN即使没训练也能改善图像！
# 原因1：卷积本身就是滤波器！
   # 卷积 = 加权平均 = 平滑
   # 卷积操作 = 局部加权平均
   # 椒盐噪音（孤立的点）会被周围的像素"平滑掉"
# 原因2：ReLU激活函数引入了非线性，有助于去除噪声！增强对比度
# 原因3：残差连接帮助保留重要特征，同时去除不必要的噪声！



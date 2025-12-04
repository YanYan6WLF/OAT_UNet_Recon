# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       toy_data.py
   Project Name:    toy_data 
   Author :         Yan
   Date:            2025/12/01
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/12/01:
-------------------------------------------------
"""
# start testing your code from small data

import numpy as np
import matplotlib.pyplot as plt
import torch
from FISTA import traditional_fista
from basicblock1 import BasicBlock1
from basicblock2 import BasicBlock2
from basicblock3 import BasicBlock3, ISTA_Net
from test_block import test_basic_block
from FISTA_NET import FISTANet,BasicBlock

def create_toy_data():
   '''
   Create toy data for testing.
   '''

   img_size=64
   img=np.zeros((img_size, img_size))

   center=img_size//2
   radius=15
   for i in range(img_size):
         for j in range(img_size):
            if (i-center)**2+(j-center)**2 < radius**2:
                  img[i,j]=1.0
   
   return img

def create_simple_projection(img_size=64,num_projections=6000):
    
   '''
   Create simple projection data from the image.
   '''

   torch.manual_seed(42)

   n_pixels=img_size*img_size
   phi=np.random.randn(num_projections,n_pixels)/np.sqrt(n_pixels)  # normalize
   phi=torch.FloatTensor(phi)
   return phi

def generate_measurement(phi, img):
   '''
   Generate measurement data from the image and projection matrix.
   b=A*x+noise
   '''
   # img_vector=img.flatten()   flatten numpy array to 1D
   img_vector=torch.FloatTensor(img).view(-1,1) # [4096,1]
   b=torch.matmul(phi, img_vector)
   return b

toy_img=create_toy_data()
phi=create_simple_projection()
b=generate_measurement(phi, toy_img)

print(f"img size: {toy_img.shape}   phi size: {phi.shape}   b size: {b.shape}")

plt.subplot(1,2,1)
plt.imshow(toy_img, cmap='gray')
plt.title('Toy Image')
plt.subplot(1,2,2)
plt.plot(b.numpy()) # torch to numpy
plt.title('Measurement b')
plt.show()

# -----------------------------------------------------------------------------

# results=traditional_fista(phi,b, num_iters=100, lambda_step=0.1, threshold=0)  #  learning rate=0.5,  loss = inf , gradient expodes
# fig, axes = plt.subplots(2,5,figsize=(15,6))
# for i, ax,in enumerate(axes.flat):
#    ax.imshow(results[10*i+9],cmap='gray')
#    ax.set_title(f"iter:{i}")
#    ax.axis('off')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(b.numpy(),color='blue')  # plot 线图
# plt.plot((phi@torch.FloatTensor(results[-1]).view(-1,1)).numpy(), color='pink',linestyle='--') # plot 线图
# plt.show()

# -----------------------------------------------------------------------------

# block1=BasicBlock1() # model
# xo=torch.zeros(1,1,64,64)  

# x_out=block1(xo, phi,b)  
# print(f"input size :{ xo.shape[2]}, output size : { x_out.shape[2]}, lambda_step: {block1.lambda_step.item():.4f}") #  .item() to get the value of a one-element tensor
# plt.imshow(x_out.detach().numpy().squeeze(), cmap='gray') #! detach from computation graph
# plt.show()

# -----------------------------------------------------------------------------


# block2=BasicBlock2()
# x0=torch.zeros(1,1,64,64)

# x_out=block2(x0,phi,b)
# print(f"input size :{ x0.shape[2]}, output size : { x_out.shape[2]}, lambda_step: {block2.lambda_step.item():.4f}, softthreshold: {block2.softthreshold.item():.4f}")
# plt.imshow(x_out.detach().numpy().squeeze(), cmap='gray' )
# plt.show()

# -----------------------------------------------------------------------------


# block3= BasicBlock3()

# results=test_basic_block(block3, phi=phi, b=b , num_iters=5) 

# # 是的，CNN即使没训练也能改善图像！
# # 原因1：卷积本身就是滤波器！
#    # 卷积 = 加权平均 = 平滑
#    # 卷积操作 = 局部加权平均
#    # 椒盐噪音（孤立的点）会被周围的像素"平滑掉"
# # 原因2：ReLU激活函数引入了非线性，有助于去除噪声！增强对比度
# # 原因3：残差连接帮助保留重要特征，同时去除不必要的噪声！

# -----------------------------------------------------------------------------

# net=ISTA_Net()
# x0=torch.zeros(1,1,64,64)
# intermediates,x=net(phi=phi,b=b, x0=x0)
# fig, axes = plt.subplots(1, len(intermediates), figsize=(15,5))
# for i, ax,in enumerate(axes.flat):
#    ax.imshow(intermediates[i].squeeze(),cmap='gray')
#    ax.set_title(f"iter:{i}")
#    ax.axis('off')
# plt.tight_layout()
# plt.show()

# -----------------------------------------------------------------------------
NET=FISTANet(num_layers=15)
X0=torch.zeros(1,1,64,64)
phi = phi.unsqueeze(0).unsqueeze(0)
b = b.unsqueeze(0).unsqueeze(0)
xnews,layers_sym, layers_st=NET(phi,b, X0)  # add batch dimension
fig, axes = plt.subplots(1, len(xnews), figsize=(15,5))
for i, ax,in enumerate(axes.flat):
   ax.imshow(xnews[i].detach().numpy().squeeze(),cmap='gray')
   ax.set_title(f"iter:{i}")
   ax.axis('off')
   print(f"iteration:{i}, loss: {torch.norm(phi@xnews[i].view(-1,1)-b).item():.4f}")
plt.tight_layout()
plt.show()




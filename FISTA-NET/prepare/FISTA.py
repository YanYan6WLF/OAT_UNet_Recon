# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       FISTA.PY
   Project Name:    FISTA
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

def traditional_fista(phi, b, num_iters=10, lambda_step=0.1, threshold=0.5):
    '''
    Implement the traditional FISTA algorithm here.
    '''
    img_size=64
    n_pixels=img_size*img_size
    x = torch.zeros((n_pixels, 1)) # initial guess
    y=x.clone()
    t=1.0

    phiTphi=torch.matmul(phi.t(),phi)
    phiTb= phi.t() @ b

    results=[] # Save intermediate results

    for i in range(num_iters):

        x_new=y-lambda_step*phiTphi@y+ lambda_step*phiTb
        x_new=torch.sign(x_new)*torch.relu((torch.abs(x_new)-threshold)) # Relu sparsity smaller than threshold set to zero
        t_new=(1+np.sqrt(1+4*t**2))/2
        y=x_new+((t-1)/t_new)*(x_new-x)
        
        x=x_new
        t=t_new

        results.append(x.view(img_size,img_size).detach().numpy())

        print(f"iteration:{i}, loss: {torch.norm(phi@x-b).item():.4f}")

    return results


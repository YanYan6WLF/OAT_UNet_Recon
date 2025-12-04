import torch
import numpy as np
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

def compute_measure(target,outcome,data_range):
    psnr=compute_PSNR(target,outcome,data_range)
    ssim=compute_SSIM(target,outcome,data_range)
    rmse=compute_RMSE(target,outcome)
    return [psnr,ssim,rmse]

def compute_MSE(target, outcome):
    return ((target-outcome)**2).mean()

def compute_PSNR(target,outcome, data_range): # data_range: max
    if type(outcome) == torch.Tensor:
        mse=compute_MSE(target,outcome)
        psnr=10*torch.log10((data_range**2)/ mse).item()      
    else:
        mse=compute_MSE(target,outcome) 
        psnr=10*np.log10((data_range**2)/ mse)      
    return psnr

def compute_RMSE(target,outcome):
    if type(outcome) == torch.Tensor:
        rmse = torch.sqrt(compute_MSE(target,outcome)).item()
    else:
        rmse = np.sqrt(compute_MSE(target,outcome))   
    return rmse

def gaussian(window_size,sigma):
    gauss = torch.Tensor([exp(-(x-window_size//2)**2 / float(2*sigma **2)) for x in range(window_size)]) # exp(- (x - μ)^2 / (2σ^2))
    return gauss/gauss.sum() # SSIM uses it as a weighted average window.To ensure that sum(weights) = 1, it is a true "average".

def create_window(window_size,channel):
    _1d_win=gaussian(window_size,1.5).unsqueeze(1) # [0.01, 0.05, 0.12, 0.20, 0.25, 0.28, 0.25, 0.20, ...] shape =[11] → # shape: [11.1]
    _2d_win=_1d_win.mm(_1d_win.t()),float().unsqueeze(0).unsqueeze(0)  # [11,1] @ [1,11] → [11,11] 2D guassian kernel
    window=Variable(_2d_win.expand(channel,1,window_size,window_size).contiguous()) # [ out_channels, in_channels, H, W ] ： kernel in pytorch must have 4 dimensions
    return window

def compute_SSIM(target, outcome, data_range, window_size=11, channel=1, size_average=True):
    if len(target.size()) ==2:
        shape=target.shape[-1]
        target=target.view(1,1,shape,shape)
        outcome=outcome.view(1,1,shape,shape)
    window=create_window(window_size, channel)
    window= window.type_as(target)

    mu1=F.conv2d(target,window,padding=window_size//2) # keep the same dimension; average
    mu2=F.conv2d(outcome,window,padding=window_size//2)
    mu1_sq,mu2_sq = mu1.pow(2), mu2.pow(2) # σ1² = E[x1²] - (E[x1])² intra; variance
    mu1_mu2=mu1*mu2 # σ12 = E[x1*x2] − E[x1] E[x2] inter; covariance

    sigma1_sq=F.conv2d(target*target,window,padding=window_size//2) - mu1_sq
    sigma2_sq=F.conv2d(outcome*outcome, window,padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(target*outcome, window, padding= window_size//2) - mu1_mu2

    # avoid division by zero
    C1,C2 = (0.01*data_range)**2, (0.03*data_range)**2
    ssim= ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq + sigma2_sq +C2)) # luminance term: (2μ1μ2 + C1) / (μ1² + μ2² + C1) # contrast term: (2μ1μ2 + C1) / (μ1² + μ2² + C1)
    # ssim_map shape = [1, 1, H, W]

    if size_average:
        return ssim.mean().item() # ↓mean over all dims↓ [1,1,H,W]  →  [scalar]

    else:
        return ssim.mean(1).mean(1).mean(1).item() # [1, 1, H, W] → [1, H, W]; [1, H, W] → [1, W]; [1, W] → [1] scalar



    






# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       solver.py
   Project Name:    Solver
   Author :         Yan
   Date:            2025/12/02
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/12/02:
-------------------------------------------------
"""

'''
Solver class = Coaching system
Inputs:
- model: the network to be trained
- criterion: loss function
- optimizer: optimization algorithm 
- device: computation device (CPU or GPU)
- train_loader: DataLoader for training data
- test_loader: DataLoader for testing data



ARGS:
- save_path: path to save the trained model 
- num_epochs: number of training epochs
'''
import argparse
import torch 
import torch.nn as nn
from torch.nn import functional as F
import visualization as vis
from Hyper_parameters import get_args
from DataLoader import get_dataset # train_loader test_loader
import os 
import numpy as np
from AverageMeter import AverageMeter
import time
import logging.config
# from scipy.sparse import csc_matrix
from FISTA_NET import FISTANet
from metric import compute_measure
import imageio 



# ---------------------------------------------------------------------- Solver -----------------------------------------------------------------------------
class solver(object):
   # 定义一个叫 Solver 的类
   # 类 = 一个工具箱，里面有各种工具（函数）
   # object = 所有类的祖宗（Python的基础类）
   def __init__(self, model, trainloader, validateloader,testloader):
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      print(f"Using device: {self.device}")
      if torch.cuda.is_available():
            print(torch.cuda.is_available())
            print(torch.cuda.get_device_name(0))
      
      self.model =model # has been loaded to device before input
      self.train_loss=nn.MSELoss()

      self.args=get_args()
      self.train_loader=trainloader
      self.validate_loader=validateloader
      self.test_loader=testloader


      # ----------------------------- For Visualization & Record-----------------------------
      self.lrs=[[] for _ in range(7)]
      self.wbs=[]
      self.test_ssim=[]
      self.test_psnr=[]
      self.test_rmse=[]
      self.validate_ssim=[]
      self.validate_psnr=[]
      self.validate_rmse=[]
      self.train_losses=[]
      self.validate_losses=[]

      self.train_losses_Meter = AverageMeter()
      self.validate_losses_Meter = AverageMeter()
      self.epoch_time = AverageMeter()
      self.validate_ssim_meter = AverageMeter()
      self.validate_psnr_meter = AverageMeter()
      self.validate_rmse_meter = AverageMeter()

      self.test_ssim_meter = AverageMeter()
      self.test_psnr_meter = AverageMeter()
      self.test_rmse_meter = AverageMeter()

      

      logging.config.fileConfig("./logging.conf")
      log = logging.getLogger()

      # ------- set different ir for regularization weights and network weights ----------------
      self.optimizer= torch.optim.Adam(
         [# [dict] list support to optimize differnt parameter with different lr
         # the following is torch.optimizer.param_group
            {'params': self.model.fcs.parameters()},# dropout ???? default value
            {'params': self.model.w_mu,    'lr': 0.001},
            {'params': self.model.b_mu,    'lr': 0.001},
            {'params': self.model.w_theta, 'lr': 0.001},
            {'params': self.model.b_theta, 'lr': 0.001},
            {'params': self.model.w_rho,   'lr': 0.001},
            {'params': self.model.b_rho,   'lr': 0.001},
         ],
         lr=self.lr, weight_decay=self.args.weight_dacay)                 # weight decay: L2 regularization weight_decay（权重衰减）就是训练时每一步都轻轻压一下参数，让它们不要长得太大。这样模型更健康、不容易过拟合、训练更稳定。
        #  self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.args.step_size, gamma=self.args.gamma) # every 10 epochs, multiply the learning rate by 0.9
      print(f"optimization is ready")

   # ------------------------ Trained Model Saving and Loading ---------------------------
   def save_model(self,iter):
      os.makedirs(self.args.save_model_path, exist_ok=True) # check if this file exist, if not, create a new one, if exists. do nothing
      model_file=os.path.join(self.save_model_path,'epoch_{}.ckpt'.format(iter))
      torch.save(self.model.state_dict(), model_file)
      print(f'Checkpoint saved: {model_file}')

   def load_Model(self, iter):
      model_file=os.path.join(self.save_model_path,'epoch_{}.ckpt'.format(iter))
      self.model.load_state_dict(torch.load(model_file))
      print(f'Checkpoint has been loaded: {model_file}')



   # ------------------------------- L1 Loss ------------------------------------------------
   def L1_loss(self,outcome,target):
      '''
      compute l1 loss: L1 weight is 0.1 by default
      '''
      err=self.args.L1_weight*torch.mean(torch.abs(outcome-target))
      return err

   # -------------------------------------- Training ------------------------------------------
   def train(self):
      start_time=time.time()
      train_losses=self.train_losses
      for epoch in range(self.args.start_epoch,self.args.num_epochs):
         self.model.train(True)
         print(f"Epoch: {epoch}")

         for batch_idx, batch in enumerate(self.train_loader):
            A=batch['A'].to(self.device) # [204800, 15625] shape
            b=batch['b'].to(self.device) # [400, 512]
            xold=batch['input'].to(self.device) # shape
            target=batch['target'].to(self.device) # [B,1,125,125]

            [xnews, layers_sym, layers_st] = self.model(phi=A, b=b, x0=xold)

            # loss ssim psnr
            loss =self.loss_formula(self,xnews, target,layers_sym,layers_st)

            self.model.zero_grad() # eliminate former gradient calculation
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            vis.save_lrs(lrs=self.lrs, param_groups=self.optimizer.param_groups)
            vis.save_wbs(param_groups=self.optimizer.param_groups,wbs=self.wbs)
            self.scheduler.step()

            self.train_losses_Meter.update(loss.item(), target.shape[0])
            train_losses.append({'Epoch':epoch,'batch_idx':batch_idx,'loss':loss.item()})

            # Print Process
            if batch_idx % self.args.step_size == 0:
               print(
                  f'Train Epoch: {epoch}'
                  f'[{batch_idx * target.shape[0]} / {len(self.train_loader.dataset)}' # the total sample quantity in train_loader
                  f'({100 * batch_idx / len(self.train_loader):.0f}%)]\t'
                  f"Batch loss: {loss.item():.6f}\t"
                  f"Batch Time: {time.time()-start_time:.1f}s"
                  )

               print('Gradient value w_mu: {}'.format(self.model.w_mu))
               print('Gradient value b_mu: {}'.format(self.model.b_mu))
               print('Threshold value w_theta: {}'.format(self.model.w_theta))
               print('Threshold value b_theta: {}'.format(self.model.b_theta))
               print('Two step update value w_rho: {}'.format(self.model.w_rho))
               print('Two step update value b_rho: {}'.format(self.model.b_rho))

         if epoch % 1==0:
            self.save_model(epoch)
         # ----------------------------------------- Validation --------------------------------------------------
         if epoch % 10 == 0:
            self.model.eval()
            validate_losses=self.validate_losses
            print("Validation Starts")
            with torch.no_grad():
               for batch_id, batch in  enumerate(self.validate_loader):
                  A=batch['A'].to(self.device) # [204800, 15625] shape
                  b=batch['b'].to(self.device) # [400, 512]
                  xold=batch['input'].to(self.device) # shape
                  target=batch['target'].to(self.device) # [B,1,125,125]
                  [xnews, layers_sym, layers_st] = self.model(phi=A, b=b, x0=xold)
                  # loss ssim psnr
                  loss =self.loss_formula(self,xnews, target,layers_sym,layers_st)
                  self.validate_losses_Meter.update(loss.item(), target.shape[0])
                  validate_losses.append({'Epoch':epoch,'batch_idx':batch_idx,'loss':loss.item()})

                  output=xnews[-1]
                  [psnr,ssim,rmse] = compute_measure(outcome=output, target=target, data_range=self.args.data_range) # scalar
                  self.validate_ssim.append(ssim)
                  self.validate_psnr.append(psnr)
                  self.validate_rmse.append(rmse)
                  self.validate_ssim_meter.update(ssim.item(), target.shape[0])
                  self.validate_psnr_meter.update(psnr.item(), target.shape[0])
                  self.validate_rmse_meter.update(rmse.item(), target.shape[0])

                  # --- save results ---
                  self.save_results(output=output,target=target,batch_idx=batch_id,path=self.args.save_validation_path)
      print('Training and validation are completed!')

      
   def loss_formula(self, xnews, target, layers_sym, layers_st):
      loss_discrepancy=self.train_loss(xnews[-1],target) + self.L1_loss(outcome=xnews[-1],target=target)        # MSE penalty: square difference per pixel 
      loss_constraint=0          # This loss penalizes CNNs for deviating too far from the original mathematical structure of FISTA.
      for k, _ in enumerate(layers_sym,0): # 0: starting index
         loss_constraint += torch.mean(torch.pow(layers_sym[k],2)) # L2 regularization
      sparsity_constraint = 0
      for k, _ in enumerate(layers_st,0):
         sparsity_constraint += torch.mean(torch.abs(layers_st[k])) # L1 regularzation)
      loss = loss_discrepancy + 0.01* loss_constraint + 0.001* sparsity_constraint # scalar: mean value of all sample in one batch
      return loss

   def test(self):
      self.load_model(self.args.test_epoch) #
      self.model.eval() # close dropout BN

      with torch.no_grad():
         for batch_id, batch in enumerate(self.test_loader):
            A=batch['A'].to(self.device) # [204800, 15625] shape
            b=batch['b'].to(self.device) # [400, 512]
            xold=batch['input'].to(self.device) # shape
            target=batch['target'].to(self.device) # [B,1,125,125]
            [xnews, _, _] = self.model(phi=A, b=b, x0=xold)

            output=xnews[-1]
            [psnr,ssim,rmse] = compute_measure(outcome=output, target=target, data_range=self.args.data_range) # scalar
            self.test_ssim.append(ssim)
            self.test_psnr.append(psnr)
            self.test_rmse.append(rmse)
            self.test_ssim_meter.update(ssim.item(), target.shape[0])
            self.test_psnr_meter.update(psnr.item(), target.shape[0])
            self.test_rmse_meter.update(rmse.item(), target.shape[0])
            # --- save results ---
            self.save_results(output=output,target=target,batch_idx=batch_id,path=self.args.save_test_path)
         return xnews


   def save_results(self,output,target,batch_idx,path):

      os.makedirs(path, exist_ok=True) # check if this file exist, if not, create a new one, if exists. do nothing
      output =output.cpu().numpy()
      target=target.cpu().numpy()
      B=target.shape[0]
      for i in range(B):
         out_img=output[i,0,:,:]
         tar_img=target[i,0,:,:]
         out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min() + 1e-8)
         tar_img = (tar_img - tar_img.min()) / (tar_img.max() - tar_img.min() + 1e-8)
         combined= np.concatenate((out_img,tar_img),axis=1)
         file=os.path.join(path,f"batch{batch_idx}_img{i}.png")
         imageio.imwrite(file,combined)
      print(f'Checkpoint saved: {file}')


# ---------------------------------------------- Visualization -----------------------------------------------------

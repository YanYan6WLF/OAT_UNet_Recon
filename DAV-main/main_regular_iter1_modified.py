# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main_regular_iter1_modified.py
   Project Name:    project_name
   Author :         Yan
   Date:            2025/11/08
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/11/08:
-------------------------------------------------
"""

# ==============================================================
# Qestions
# ==============================================================
# k or K+1
# Visualizer
# xk=xk_mat
# 每层？？  losses.update(loss.item(), rawdata.size(0))


import logging.config
from tqdm import tqdm
from model_cnn_new import Reg_net
from Visualizer import Visualizer
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from   ReconDataset import ReconDataset
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.stats as st
import scipy.io as scio
import click
import argparse
import matlab.engine as eng


# ==============================================================
# 1. Parameter Setting and logging
# ==============================================================

parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--dataset_pathr', type=str, default='D:\MATLAB_Yan\Test\Mouse_6', help='path of dataset')
# parser.add_argument('--vis_env', type=str, default='model_based_iter1_regularization', help='visualization environment')
parser.add_argument('--save_path', type=str, default='checkpoint/', help='path of saved model')
parser.add_argument('--file_name', type=str, default='Reg_net_iter1_regularization.ckpt', help='file name of saved model')
parser.add_argument('--learning_rate', type=float, default=0.004, help='learning rate')
parser.add_argument('--batch_size', type=int, default=20, help='batch_size of training')
parser.add_argument('--test_batch', type=int, default=5, help='batch_size of testing')
# parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--loadcp', type=bool, default=False, help='if load model')
parser.add_argument('--num_epochs', type=int, default=300, help='the number of epoches')
parser.add_argument('--num_stages', type=int, default=10, help="number of total iterations")
args = parser.parse_args()

# create logger
logging.config.fileConfig("./logging.conf")
log = logging.getLogger()


# ==============================================================
# 2. Tool Function 
# ==============================================================
def calc_confidence_interval(samples, confidence_value=0.95):
    # samples should be a numpy array
    if type(samples) is list:
        samples = np.asarray(samples)
    assert isinstance(samples, np.ndarray), 'args: samples {} should be np.array'.format(samples)
    # print('Results List:', samples)
    stat_accu = st.t.interval(confidence_value, len(samples) - 1, loc=np.mean(samples), scale=st.sem(samples))
    center = (stat_accu[0] + stat_accu[1]) / 2
    deviation = (stat_accu[1] - stat_accu[0]) / 2
    return center, deviation

# Record and update the current value, average, sum, count, Top-N results, etc. of a certain indicator in real time.
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, num_top=10):
        self.reset()
        _array = np.zeros(shape=(num_top)) + 0.01
        self.top_list = _array.tolist()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def top_update_calc(self, val):
        # update the lowest or NOT
        if val > self.top_list[0]:
            self.top_list[0] = val 
            # [lowest, ..., highest]
            self.top_list.sort()
        # update mean, deviation
        mean, deviation = calc_confidence_interval(self.top_list)
        best = self.top_list[-1]
        return mean, deviation, best

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ==============================================================
# 3. MATLAB Enginering
# ==============================================================
print("Starting MATLAB engine...")
eng = matlab.engine.start_matlab()
print("MATLAB Engine started successfully")


# ==============================================================
# 4. Main Workflow
# ==============================================================


def main():

    # ------------ devide ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")



    # ------------ Initalization the Structure ------------

    model = Reg_net(1) # question
    model = nn.DataParallel(model).to(device) # question
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rates) # parameter
    criterionMSE = nn.MSELoss()

    # ------------ Dataset Loading ------------
    train_dataset = ReconDataset(args.dataset_pathr, select=True,iter=k)
    test_dataset = ReconDataset(args.dataset_pathr, select=False,iter=k)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True) # parameter
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= args.test_batch, shuffle=True)
    
    # ------------ Training Recording ------------
    losses = AverageMeter()
    total_loss = AverageMeter() # question
    batch_time = AverageMeter()
    layer_time = AverageMeter()
    train_ssim_meter = AverageMeter()
    train_psnr_meter = AverageMeter()
    # train_ssim_top20 = AverageMeter(num_top=20)
    # train_psnr_top20 = AverageMeter(num_top=20)
    test_ssim_meter = AverageMeter()
    test_psnr_meter = AverageMeter()
    # test_ssim_top10 = AverageMeter(num_top=10)
    # test_psnr_top10 = AverageMeter(num_top=10)
    vis = Visualizer(env=args.vis_env,port=5167)

    # ------------ Checkpoint ------------
    if  args.loadcp:
        checkpoint = torch.load(f"{args.save_path}stage_{k}_best_{args.file_name}")
        # model.load_state_dict(checkpoint['state_dict'])
        # start_epoch = checkpoint['epoch'] - 1
        # curr_lr = checkpoint['curr_lr']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # start_epoch = checkpoint['epoch']
        # print('%s%d' % ('training from epoch:' , start_epoch))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        args.learning_rate = checkpoint['curr_lr']
        xk=checkpoint['last_output']
        xk_np = xk.numpy() if 'torch' in str(type(xk)) else xk
        xk_mat = eng.double(xk_np.tolist())


        checkpoint0 = torch.load(f"{args.save_path}stage_{k-1}_best_{args.file_name}")
        xkm1=checkpoint0['last_output']

    
    if k==0:
        xk = torch.zeros_like(rawdata, device=device)
        xkm1=xk
        xk_mat = matlab.double(np.zeros(rawdata.shape).tolist())



    cudnn.benchmark = True

    total_step = len(train_loader)

    best_metric = {'test_ssim': 0, 'test_psnr': 0}
    log.info('train image num: {}'.format(train_dataset.__len__()))
    log.info('val image num: {}'.format(test_dataset.__len__()))

    end = time.time()

    # ------------ Epoch Loops ------------
    for epoch in range(args.start_epoch,  args.num_epochs):
        # ------------ Batch Loops ------------
        for batch_idx, (rawdata, lsqr_data,A,b) in enumerate((train_loader)): 
            # ------------ Layer Iterations ------------
            for k in range(args.num_stages): # parameter
                print(f"=============== Stage {k+1}/{args.num_stages} ===============")
                rawdata = rawdata.to(device)
                gtLSQR = lsqr_data[k].to(device)
                # A,b python numpy → matlab double
                A_mat=eng.double(A.tolist())
                b_mat=eng.double(b.tolist())
                gradup = eng.eval("A'*(A*xk - b)", nargout=1, A=A_mat, b=b_mat, xk=xk_mat) # question   
                # forward
                outputs = model(rawdata,xk,gradup,xkm1,t=1.0)
                # Backward and optimize
                loss = 20 *criterionMSE(outputs, gtLSQR) # can be changed
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), rawdata.size(0)) # AverageMeter.update(self, val, n=batch_size):

                ssim = compare_ssim(np.array(outputs.cpu().detach().squeeze()), np.array(gtLSQR.cpu().detach().squeeze()))
                train_ssim_meter.update(ssim)
                psnr = compare_psnr(np.array(outputs.cpu().detach().squeeze()), np.array(gtLSQR.cpu().detach().squeeze()),
                                    data_range=255)
                train_psnr_meter.update(psnr)

                layer_time.update(time.time() - layer_end)
                layer_end = time.time()

                log.info(
                    'Epoch [{}], Start [{}], Step [{}/{}], Layer [{}/{args.num_stages}] Loss: {:.4f}, Time [{batch_time.val:.3f}({batch_time.avg:.3f})]'
                        .format(epoch + 1, args.start_epoch, batch_idx + 1, total_step, k, args.num_stages, loss.item(),
                                batch_time=batch_time))       # question: args.num_stages+1????
             
            batch_time.update(time.time() - batch_end)
            batch_end = time.time()

        log.info(
            'Epoch [{}], Start [{}], Step [{}/{}], Loss: {:.4f}, Time [{batch_time.val:.3f}({batch_time.avg:.3f})]'
                .format(epoch + 1, args.start_epoch, batch_idx + 1, total_step, loss.item(),
                        batch_time=batch_time))
        vis.plot_multi_win(
            dict(
                losses_total=losses.val,
                losses = losses.avg
            ))

        vis.plot_multi_win(dict(train_ssim=train_ssim_meter.avg, train_psnr=train_psnr_meter.avg))
        log.info('tain_ssim: {}, train_psnr: {}'.format(train_ssim_meter.avg, train_psnr_meter.avg))

        # Validata
        if epoch % 5 == 0:
                with torch.no_grad(): 
                    for batch_idx, (rawdata, lsqr_data) in enumerate((test_loader)): 
                        rawdata = rawdata.to(device)
                        out_data = out_data.to(device)
                        gradup=eng.x_grad(xk_mat,A,b)
                        gradup=torch.from_numpy(np.array(gradup)).float().squeeze().to(device)   
                        outputs = model(rawdata,xk,gradup,xkm1,t=1.0)

                        ssim = compare_ssim(np.array(outputs.cpu().squeeze()), np.array(out_data.cpu().squeeze()))
                        test_ssim_meter.update(ssim)
                        psnr = compare_psnr(np.array(outputs.cpu().squeeze()), np.array(out_data.cpu().squeeze()),
                                            data_range=255)
                        test_psnr_meter.update(psnr)
                
                        if (batch_idx +1) % 2 ==0:
                            output_img = outputs.detach()
                            gt_image = out_data.detach()
                            input_img = rawdata.detach()
                            grad_img = gradup.detach()
                            vis.img(name='test_input',img_=15*input_img[0])                               
                            vis.img(name='test_gt',img_=15*gt_image[0])
                            vis.img(name='test_output', img_=15*output_img[0])
                            vis.img(name='test_grad', img_=15*grad_img[0])


                    vis.plot_multi_win(dict(
                        test_ssim=test_ssim_meter.avg,
                        test_psnr=test_psnr_meter.avg))
                    log.info('test_ssim: {}, test_psnr: {}'.format(test_ssim_meter.avg, test_psnr_meter.avg))

        # Decay learning rate
        if (epoch + 1) % 50 == 0:
            args.learning_rate /= 2
            update_lr(optimizer, args.learning_rate)

        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer,
                    'curr_lr': args.learning_rate,
                    },
                    args.save_path+'latest_'+args.file_name
                    )

        if best_metric['test_ssim'] < test_ssim_meter.avg:
            torch.save({'epoch': epoch,
                        'model': model,
                        'optimizer': optimizer,
                        'curr_lr': args.learning_rate,
                        },
                        args.save_path+'best_'+args.file_name
                        )
            best_metric['test_ssim'] = test_ssim_meter.avg
            best_metric['test_psnr'] = test_psnr_meter.avg
        log.info('best_ssim: {}, best_psnr: {}'.format(best_metric['test_ssim'], best_metric['test_psnr']))
    
    # save model of each layer iteration
    torch.save({
        'epoch': args.num_epochs,
        'model': model,
        'optimizer': optimizer,
        'curr_lr': args.learning_rate,
        'last_output': outputs.detach().cpu(),
    }, f"{args.save_path}stage_{k}_best_{args.file_name}")

    print(f"Saved model for Stage {k+1}: {args.save_path}stage_{k}_best_{args.file_name}")


if __name__ == '__main__':
    main()

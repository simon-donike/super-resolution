import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn.functional import normalize
from utils.ssimclass import ssim
import utils.helper_functions


import os
import sys
import math
import torch
import numpy as np
import cv2
    
    
def min_max(a,b):
    a = normalize(a, p=2.0)
    b = normalize(b, p=2.0)
    
    #a = a*255
    #b = b*255
    
    #a = a.type(torch.DoubleTensor)
    #b = a.type(torch.DoubleTensor)
    return(a,b)
    

def loss_lpips(a,b):
    # WORKS
    import lpips
    #print(a)
    # set up loss net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpips_alex = lpips.LPIPS(net='alex',verbose=False)
    lpips_alex.to(device)
    # calc loss
    res = lpips_alex(a,b)
    
    loss = torch.mean(res)
    return(loss)

def loss_mse(a,b):
    # WORKS
    return(torch.nn.functional.mse_loss(a,b))

def loss_mae(a,b):
    # WORKS
    return(torch.nn.functional.l1_loss(a,b))
    
def loss_mae_lpips(a,b,weight_mae=0.1,weight_lpips=0.9):
    # handle weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_mae, weight_lpips = torch.tensor(weight_mae),torch.tensor(weight_lpips)
    weight_mae = weight_mae.to(device)
    weight_lpips = weight_lpips.to(device)
    
    # calc losses
    mae = loss_mae(a,b)
    lpips = loss_lpips(a,b)
    # weighted average
    # (x1w1 + x2w2) / w1+w2 
    f = torch.divide(torch.add(torch.mul(mae,weight_mae), torch.mul(lpips,weight_lpips))  , torch.add(weight_mae,weight_lpips))
    #print("MAE:",mae.item(),", LPIPS:",lpips.item(),",Final:",f.item())
    return(f)

def loss_psnr(a, b):
    # orig, pred
    # WORKS, for range 0..1
    # https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
    mse = torch.mean((a - b) ** 2)
    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse))
    # revert to optimize minimum psnr
    # inspired by https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/psnr.html
    return(psnr_val)


def loss_ssim(a,b):
    # WORKS
    # https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/ssim.html
    return(ssim(a,b,window_size=11)) # 0-bad, 1-good
    

def calculate_metrics(hr,lr,sr):
    import utils.helper_functions as helper_functions
    lpips = loss_lpips(hr,sr).item()
    psnr  = loss_psnr(hr,sr).item()
    ssim  = loss_ssim(hr,sr).item()
    mae   = loss_mae(hr,sr).item()
    
    interpolated = helper_functions.interpolate_tensor(lr,300)
    interpolated = interpolated[None] # add batch dimension to tensor
    lpips_int = loss_lpips(hr,interpolated).item()
    psnr_int  = loss_psnr(hr,interpolated).item()
    ssim_int  = loss_ssim(hr,interpolated).item()
    mae_int   = loss_mae(hr,interpolated).item()
    
    ls = round_to_whole = [round(num,3) for num in [lpips,psnr,ssim,mae,lpips_int,psnr_int,ssim_int,mae_int]]
    return(ls)
    
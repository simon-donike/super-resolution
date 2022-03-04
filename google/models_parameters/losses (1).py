import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn.functional import normalize
from models_parameters.ssimclass import ssim



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
    return(res)

def loss_mse(a,b):
    # WORKS
    return(torch.nn.functional.mse_loss(a,b))

def loss_mae(a,b):
    # WORKS
    return(torch.nn.functional.l1_loss(a,b))

def loss_psnr(a, b):
    # WORKS, check for value range
    # https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
    mse = torch.mean((a - b) ** 2)
    psnr_val = 20 * torch.log10(255.0 / torch.sqrt(mse))
    # revert to optimize minimum psnr
    # inspired by https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/psnr.html
    return(-1*psnr_val)


def loss_ssim(a,b):
    # WORKS
    # https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/ssim.html
    return(1-ssim(a,b,window_size=11)) # revert to optimize minimum ssim
    
    
    


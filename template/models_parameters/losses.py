import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn.functional import normalize
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from skimage.metrics import structural_similarity as ssim



import os
import sys
import math
import torch
import numpy as np
import cv2

    
"""
def min_max(a,b):
    
    #returns 0..1 normlaized data
    
    mins = a.min(dim=1, keepdim=True)
    maxs = a.max(dim=1, keepdim=True)
    a = (a - mins) / (maxs - mins)
    
    mins = b.min(dim=1, keepdim=True)
    maxs = b.max(dim=1, keepdim=True)
    b = (b - mins) / (maxs - mins)
    return(a,b)
"""
    
    
    
def min_max(a,b):
    a = normalize(a, p=2.0)
    b = normalize(b, p=2.0)
    
    #a = a*255
    #b = b*255
    
    #a = a.type(torch.DoubleTensor)
    #b = a.type(torch.DoubleTensor)
    return(a,b)
    
    
def loss_lpips(a,b):
    # DOENST WORK
    import lpips
    # transform to double
    a = a.type(torch.DoubleTensor)
    b = b.type(torch.DoubleTensor)
    print(a)
    # set up loss net
    lpips_alex = lpips.LPIPS(net='alex',verbose=False)
    # calc loss
    res = lpips_alex(a,b)
    return(res)

def loss_mse(a,b):
    # WORKS
    loss = nn.MSELoss()
    mse = loss(a, b)
    mse = Variable(mse.data, requires_grad=True)
    return(mse)

def loss_mae(a,b):
    # WORKS
    loss = nn.L1Loss()
    mae = loss(a, b)
    mae = Variable(mae.data, requires_grad=True)
    return(mae)

def loss_psnr(a, b):
    # WORKS, check for value range
    # https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
    mse = torch.mean((a - b) ** 2)
    psnr_val = 20 * torch.log10(255.0 / torch.sqrt(mse))
    return(100-psnr_val) # revert to optimize minimum psnr

"""
def loss_ssim(a,b):
    # to numpy, not possible for batches>1
    a = a.detach().numpy()[0] # detach from grad and to numpy
    b = b.detach().numpy()[0] # detach from grad and to numpy
    a = np.transpose(a,(1,2,0)) # transform to numpy image format
    b = np.transpose(b,(1,2,0)) # transform to numpy image format
    ssim_const = ssim(a, b,data_range=b.max() - b.min(),multichannel=True) # calc with ssim
    ssim_const = torch.tensor(ssim_const) # transform to tensor
    ssim_const = Variable(ssim_const.data, requires_grad=True) # back to torch with grad
    return(1-ssim_const) # revert to optimize minimum ssim
"""

def loss_ssim(a,b):
    # https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/ssim.html
    from models_parameters.ssimclass import ssim
    return(ssim(a,b))
    
    
    


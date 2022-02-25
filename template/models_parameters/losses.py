import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn.functional import normalize
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
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
    # MSE
    loss = nn.MSELoss()
    mse = loss(a, b)
    mse = Variable(mse.data, requires_grad=True)
    return(mse)

def loss_mae(a,b):
    # MSE
    loss = nn.L1Loss()
    mae = loss(a, b)
    mae = Variable(mae.data, requires_grad=True)
    return(mae)

def loss_psnr(a, b):
    # https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
    mse = torch.mean((a - b) ** 2)
    return(20 * torch.log10(255.0 / torch.sqrt(mse)))

def loss_sim1(a,b):
    # to numpy, not possible for batches>1
    a = a.detach().numpy()[0]
    b = b.detach().numpy()[0]
    ssim_const = ssim(a, b,data_range=b.max() - b.min(),multichannel=True)
    # ValueError: win_size exceeds image extent.  If the input is a multichannel (color) image, set multichannel=True ?!
    return(ssim_const)


def loss_ssim2(img1, img2):
    # https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
    img1,img2 = min_max(img1,img2)
    print(img1)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def loss_ssim3(a,b):
    # https://github.com/VainF/pytorch-msssim
    ssim_loss = 1 - ssim( a, b, data_range=1, size_average=True,multichannel=True) # return a scalar
    return(ssim_loss)
    # win_size exceeds image extent.  If the input is a multichannel (color) image, set multichannel=True
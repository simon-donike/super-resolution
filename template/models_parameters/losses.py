import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable


def loss_psnr(a, b):
    # PSNR
    # https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/#:~:text=Peak%20signal%2Dto%2Dnoise%20ratio%20(PSNR)%20is%20the,with%20the%20maximum%20possible%20power.
    # PSNR = 20*log(max(max(f)))/((MSE)^0.5)
    # result in db
    # lower->better
    try:
        import math
        mse = np.mean((a - b) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal. Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return(psnr)
    except ValueError:
        return(0)
        

def loss_mae(a,b):
    # MAE
    error = torch.abs(a - b).sum().data
    error = Variable(error.data, requires_grad=True)
    return(error)


def loss_mse(a,b):
    # MSE
    loss = nn.MSELoss()
    mse = loss(a, b)
    mse = Variable(mse.data, requires_grad=True)
    return(mse)
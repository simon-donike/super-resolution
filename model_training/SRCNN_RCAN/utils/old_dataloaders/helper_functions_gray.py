import numpy as np
import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable
import cv2

def minmax(img):
  # TODO: hist eq.
  return(img-np.min(img,axis=(0,1)) ) / (np.max(img,axis=(0,1))-np.min(img,axis=(0,1)))
  

def interpolate(img,size=300):
    """
    Input:
        - Image
    Output:
        - Image upsampled 
    """
    dim = (size, size)
    b1 = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    
    return(b1)


def plot_tensors(a,b,c,loss):
    
  a = a.cpu().detach().numpy()[0]
  b = b.cpu().detach().numpy()[0]
  c = c.cpu().detach().numpy()[0]

  a,b,c = minmax(a),minmax(b),minmax(c)

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,10))
  ax1.imshow(a)
  ax1.set_title("Ground Truth - Spot6")
  ax2.imshow(b)
  ax2.set_title("X - Sen2")
  ax3.imshow(c)
  ax3.set_title("Pred - Sen2\nLoss: "+str(loss.item()))
  plt.show()
 
"""
def psnr(a, b):
    # a = original, b = compressed
    # https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/#:~:text=Peak%20signal%2Dto%2Dnoise%20ratio%20(PSNR)%20is%20the,with%20the%20maximum%20possible%20power
    # calculation between np arrays
    try:
        import math
        mse = np.mean((a - b) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal. Therefore PSNR have no importance.
            return 100
        max_pixel = 1.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return(round(psnr,2))
    except ValueError:
        return(0)
"""
def interpolate_tensor(t,size=300):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = t.cpu().detach().numpy()[0]
    #t = np.transpose(t,(1,2,0))
    t = interpolate(t,size)
    #t = np.transpose(t,(2,0,1))
    t = torch.tensor(t)
    t = t.to(device)
    return(t)
    
  
def plot_tensors_window(a,b,c,psnr_int,psnr_pred):
  
  "plot with 100x100 window extract"
  a = a.cpu().detach().numpy()[0]
  b = b.cpu().detach().numpy()[0]
  c = c.cpu().detach().numpy()[0]
  #a = np.transpose(a,(1,2,0))
  #b = np.transpose(b,(1,2,0))
  #c = np.transpose(c,(1,2,0))
  a,b,c = minmax(a),minmax(b),minmax(c)
  a = a[0]
  b = b[0]
  c = c[0]

  # upsample b
  b_i = interpolate(b,300)

  fig, axs = plt.subplots(2, 4,figsize=(15,7))
  axs[0,0].imshow(a)
  axs[0,0].set_title("Ground Truth")
  axs[0,1].imshow(b)
  axs[0,1].set_title("X")
  axs[0,2].imshow(b_i)
  axs[0,2].set_title("interpol. X\nPSNR: "+str(psnr_int))
  axs[0,3].imshow(c)
  axs[0,3].set_title("Pred\nPSNR: "+str(psnr_pred))

  axs[1,0].imshow(a[0:100,0:100,])
  axs[1,0].set_title("Ground Truth - window")
  axs[1,1].imshow(b[0:25,0:25,])
  axs[1,1].set_title("X - window")
  axs[1,2].imshow(b_i[0:100,0:100,])
  axs[1,2].set_title("interpol. X")
  axs[1,3].imshow(c[0:100,0:100,])
  axs[1,3].set_title("Pred - window")
  plt.show()
  


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
import numpy as np
import matplotlib.pyplot as plt
import torch
#from prettytable import PrettyTable
import cv2
import warnings



def minmax(img):
  return(img-np.min(img) ) / (np.max(img)-np.min(img))
def minmax_percentile(img,perc=2):
  lower = np.percentile(img,perc)
  upper = np.percentile(img,100-perc)
  img[img>upper] = upper
  img[img<lower] = lower
  return(img-np.min(img) ) / (np.max(img)-np.min(img))


def interpolate(img,size=300):
    """
    Input:
        - Image
    Output:
        - Image upsampled 
    """
    img = np.transpose(img,(2,0,1))
    dim = (size, size)
    b1 = cv2.resize(img[0], dim, interpolation = cv2.INTER_CUBIC)
    b2 = cv2.resize(img[1], dim, interpolation = cv2.INTER_CUBIC)
    b3 = cv2.resize(img[2], dim, interpolation = cv2.INTER_CUBIC)
    
    img = np.dstack((b1,b2,b3))
    img = np.transpose(img,(0,1,2))
    return(img)


def interpolate_tensor(t,size=300):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = t.cpu().detach().numpy()[0]
    t = np.transpose(t,(1,2,0))
    t = interpolate(t,size)
    t = np.transpose(t,(2,0,1))
    t = torch.tensor(t)
    t = t.to(device)
    return(t)
    
  
def plot_tensors_window(a,b,c,psnr_int,psnr_pred,fig_path="show"):
  warnings.filterwarnings("ignore")
  
  "plot with 100x100 window extract"
  a = a.cpu().detach().numpy()[0]
  b = b.cpu().detach().numpy()[0]
  c = c.cpu().detach().numpy()[0]
  a = np.transpose(a,(1,2,0))
  b = np.transpose(b,(1,2,0))
  c = np.transpose(c,(1,2,0))
  a,b,c = minmax_percentile(a),minmax_percentile(b),minmax_percentile(c)

  # upsample b
  b = interpolate(b,75)
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
  
  if fig_path=="show":
    plt.show()
  else:
    # create timestamp
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    plt.savefig(fig_path+dt_string+"_val_PSNR_"+str(psnr_pred)+".png")


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
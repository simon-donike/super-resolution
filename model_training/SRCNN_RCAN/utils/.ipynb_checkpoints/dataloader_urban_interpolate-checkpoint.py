import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import cv2
import skimage
from skimage import exposure
import rasterio





class Dataset(Dataset):
    def __init__(self):
        self.location = "C:\\Users\\accou\\Documents\\GitHub\\SelfExSR\\data\\BSD100\\image_SRF_4\\"
        files_list = os.listdir(self.location) # returns list
        lr,hr = [],[]
        
        for i in files_list:
            if i[14:16]=="HR":
                hr.append(self.location+i)
            if i[14:16]=="LR":
                lr.append(self.location+i)
        self.data = pd.DataFrame(zip(lr,hr), columns=['LR','HR'])
                
        
        
    def __len__(self):
        return(len(self.data))
    
    
    def interpolate(self,img,size=300):
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
        img = np.transpose(img,(2,0,1))
        return(img)

    def __getitem__(self,idx):
        # filter data
        current = self.data.iloc[idx]
        # load file from disc
        lr = rasterio.open(current["LR"]).read() / 255.0
        hr = rasterio.open(current["HR"]).read() / 255.0
        lr,hr = np.transpose(lr,(1,2,0)),np.transpose(hr,(1,2,0))
        
        #print(lr.shape,hr.shape)
        
        # resize to standard
        lr = self.interpolate(lr,75)
        lr = np.transpose(lr,(1,2,0))
        
        lr = self.interpolate(lr,300)
        hr = self.interpolate(hr,300)
        
        # transform to tensor
        lr,hr  = torch.from_numpy(lr), torch.from_numpy(hr)
        lr,hr = lr.float(), hr.float()
        return(lr,hr)
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision

import rasterio

import warnings
warnings.filterwarnings("ignore")

# Define torch dataset Class
class Dataset(Dataset):
    def __init__(self,folder_path,dataset_file,sen2_amount=1,sen2_tile="all"):
        
        # define filepaths
        self.folder_path = folder_path
        # read file
        self.df = pd.read_pickle(dataset_file)
        # set amount of sen2 pictures that should be returned
        self.sen2_amount = sen2_amount
        
        # filter for sen2 tile
        if sen2_tile!="all":
            self.df = self.df[self.df["sen2_tile"]==sen2_tile]
            
        # clear up DF
        self.df = self.df[self.df["sen2_no"]>2]
        try:
            self.df = self.df.drop(labels=["level_0"], axis=1)
        except KeyError:
            pass
        self.df = self.df.reset_index()
        
    def __len__(self):
        """
        Returns length of data
        """
        return(len(self.df))
        
    

    def rgb2gray(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    
 
    def __getitem__(self,idx):
        
        current = self.df.iloc[idx]
        spot6_file = current["spot6_filenames"]
        sen2_files = current["sen2_filenames"]
        subfolder = str(current["subfolder"])
        other_valid_acq = current["other_valid_acq"]        


        """ORDER SEN2 DATES"""
        ordered_sen2 = []
        sen2_clean = {}
        for i in sen2_files:
            sen2_clean[i[:61]] = i
        for i in sorted(other_valid_acq):
            s = other_valid_acq[i][1][:61]
            if s in sen2_clean:
                ordered_sen2.append(sen2_clean[s])
        sen2_files = ordered_sen2
        
        """READ SPOT6"""
        #GOOGLE COLAB MODE: READING FROM SUBFODLERS
        #with rasterio.open(self.folder_path+"y/"+spot6_file) as dataset:
        spot6 = rasterio.open(self.folder_path+"y_sub/"+subfolder+"/"+spot6_file).read()

    
        """READ SEN2 SERIES"""
        # read first file
        sen2 = rasterio.open(self.folder_path+"x_sub/"+subfolder+"/"+sen2_files[0]).read()
        
        if self.sen2_amount>1:
            # read following sen2 and stack
            count=1
            for sen2_file in sen2_files[1:]:
                # read file as array
                sen2_following = rasterio.open(self.folder_path+"x_sub/"+subfolder+"/"+sen2_file).read()
                # stack to previous images
                sen2 = np.concatenate([sen2, sen2_following])

                # break if all wanted files loaded
                count=count+1
                if count==self.sen2_amount:
                    break
            # if final count not yet reached, repeat last chip until enough are there
            while count<self.sen2_amount:
                sen2 = np.concatenate([sen2, sen2_following])
                count=count+1
        
        # transform to tensor
        sen2  = torch.from_numpy(sen2)
        spot6 = torch.from_numpy(spot6)
        sen2 = sen2.float()
        spot6 = spot6.float()
        
        #print(len(sen2_files),sen2.size())
        
        # define transformer
        transform_spot = transforms.Compose([transforms.Normalize(mean=[479.0, 537.0, 344.0], std=[430.0, 290.0, 229.0]) ])
        # dynamically define transform to reflect shape of tensor
        trans_mean,trans_std = [78.0, 91.0, 62.0]*self.sen2_amount,[36.0, 28.0, 30.0]*self.sen2_amount
        transform_sen = transforms.Compose([transforms.Normalize(mean=trans_mean, std= trans_std)])
        # perform transform
        sen2  = transform_sen(sen2)
        spot6 = transform_spot(spot6)
        
        
        # perform last sanity check, load random image if images arent expected shape
        while sen2.size()!= torch.Size([3* self.sen2_amount,75,75]) or spot6.size()!=torch.Size([3, 300, 300]):
            print("Warning: wrong image size in dataloader! File:",spot6_file,"or",sen2_files)
            print(sen2.size(),spot6.size())
            sen2,spot6 = self.__getitem__(random.randint(0,self.__len__()))
        
        # transform to gray
        sen2 = torchvision.transforms.functional.rgb_to_grayscale(sen2, 1)
        spot6 = torchvision.transforms.functional.rgb_to_grayscale(spot6, 1)
        
        # return result
        return(sen2,spot6)

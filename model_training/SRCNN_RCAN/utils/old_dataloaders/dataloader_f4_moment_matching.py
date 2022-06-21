import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

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
    
    def moment_matching(self,sen2,spot6):    
        """
        {[spot6 - mean(spot6)] / stdev(spot6) } * stdev(sen2) 
        + mean(sen2)
        """

        c = 0
        for channel_sen,channel_spot in zip(sen2,spot6):
            c +=1
            #calculate stats
            sen2_mean   = np.mean(channel_sen)
            spot6_mean  = np.mean(channel_spot)
            sen2_stdev  = np.std(channel_sen)
            spot6_stdev = np.std(channel_spot)
            
            # calculate moment per channel
            channel_result = (((channel_spot - spot6_mean) / spot6_stdev) * sen2_stdev) + sen2_mean
            
            # stack channels to single array
            if c==1:
                result = channel_result
            else:
                result = np.dstack((result,channel_result))
            # transpose back to Cx..
            
        result = result.transpose((2,0,1))   
        return(result)
        
        
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
        
        
        # perform moment matching
        spot6 = self.moment_matching(sen2,spot6)
        
        # stretch to 0..1
        spot6 = spot6/10000.0
        sen2  = sen2/10000.0
        
        # transform to tensor
        sen2  = torch.from_numpy(sen2)
        spot6 = torch.from_numpy(spot6)
        sen2 = sen2.float()
        spot6 = spot6.float()
        
        
        # perform last sanity check, load random image if images arent expected shape
        while sen2.size()!= torch.Size([3* self.sen2_amount,75,75]) or spot6.size()!=torch.Size([3, 300, 300]):
            print("Warning: wrong image size in dataloader! File:",spot6_file,"or",sen2_files)
            print(sen2.size(),spot6.size())
            sen2,spot6 = self.__getitem__(random.randint(0,self.__len__()))
        
        # return result
        return(sen2,spot6)
        

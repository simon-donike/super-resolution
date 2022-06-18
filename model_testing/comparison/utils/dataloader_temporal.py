import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import cv2
import skimage
from skimage import exposure
import rasterio

import warnings
warnings.filterwarnings("ignore")

# Define torch dataset Class
class Dataset(Dataset):
    def __init__(self,folder_path,dataset_file,test_train_val="train",transform="None",sen2_amount=1,sen2_tile="all",location="colab",filter_point="None"):
        
        # set on which machine the process runs
        self.location = location
        # define filepaths
        self.folder_path = folder_path
        # read file
        self.df = pd.read_pickle(dataset_file)
        # set amount of sen2 pictures that should be returned
        self.sen2_amount = sen2_amount
        # define transformer
        self.transform = transform
        # which data
        self.test_train_val = test_train_val
        
        # filter for points over notmal areas
        self.df = self.df[self.df["Code_simplified"].isin(['Wetlands', 'Agricultural Areas', 'Artificial Surfaces','Forest and seminatural Areas', 'Water Bodies'])]
        
        # filter for sen2 tile
        if sen2_tile!="all":
            self.df = self.df[self.df["sen2_tile"]==sen2_tile]
        #filter for single point 
        if filter_point!="None" and filter_point!="Area":
            self.df = self.df[(self.df.x == filter_point[0]) & (self.df.y == filter_point[1])]
            self.df.loc[self.df.index.repeat(4)]
            self.df.loc[self.df.index.repeat(4)]
        if filter_point=="Area":
            import geopandas
            print("filtering for inference Area")
            area = geopandas.read_file("utils/inference_file.geojson")
            self.df = geopandas.clip(self.df, area, keep_geom_type=True)
            
        # filter for train/test data
        if self.test_train_val == "train":
            self.df = self.df[self.df["type"]=="train"]
        if self.test_train_val == "test":
            self.df = self.df[self.df["type"]=="test"]
        if self.test_train_val == "val":
            self.df = self.df[self.df["type"]=="train"]
        # if validation required, choose 10 perc
        if self.test_train_val == "val":
            _, self.df = train_test_split(self.df, test_size=0.1)
        
        # clear up DF for invalid sen2 and/or spot6
        self.df = self.df[self.df["sen2_no"]>2]
        self.df = self.df[self.df["spot6_validity_2"]==True]
        
        
        # perform strat if wanted
        strat = False
        if strat==True:
            # get amount of second most present class
            self.df["Code_simplified"].value_counts()[1] 
            #calculate how many of most frequent class should be dropped, keep 100 more in most frequent class
            drop_amount = (self.df["Code_simplified"].value_counts()[0] - self.df["Code_simplified"].value_counts()[1])-100
            # drop stratified amount from largest dataset class
            self.df = self.df.drop(self.df[self.df['Code_simplified'] == "Agricultural Areas"].sample(n=drop_amount, random_state=42).index)
            # drop Wetlands
            self.df = self.df.drop(self.df[self.df.Code_simplified == "Wetlands"].index)
            # drop Water Bodies
            self.df = self.df.drop(self.df[self.df.Code_simplified == "Water Bodies"].index)
            # drop NaN
            self.df = self.df[self.df['Code_simplified'].notna()]
            
            
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
        
    """ DEFINE TRANSFORMERS """
    
    # Normal Standardization over whole dataset
    def standardize(self,sen2,spot6):
        transform_spot = transforms.Compose([transforms.Normalize(mean=[479.0, 537.0, 344.0], std=[430.0, 290.0, 229.0]) ])
        # dynamically define transform to reflect shape of tensor
        trans_mean,trans_std = [78.0, 91.0, 62.0]*self.sen2_amount,[36.0, 28.0, 30.0]*self.sen2_amount
        transform_sen = transforms.Compose([transforms.Normalize(mean=trans_mean, std= trans_std)])
        # perform transform
        sen2  = transform_sen(sen2)
        spot6 = transform_spot(spot6)
        return(sen2,spot6)
        
    # HISTOGRAM MATCHING
    def histogram_matching(self,sen2,spot6):
        # have to transpose so that multichannel understands the dimensions
        sen2 = np.transpose(sen2,(1,2,0))
        spot6 = np.transpose(spot6,(1,2,0))
        # https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.match_histograms
        result = exposure.match_histograms(image=spot6,reference=sen2,multichannel=True)
        result = np.transpose(result,(2,0,1))
        return(result)
        
    # MOMENT MATCHING
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

    # Interpolation for spot to spot training
    def interpolate(self,img,size=300):
        """
        Input:
            - Image
        Output:
            - Image upsampled 
        """
        dim = (size, size)
        b1 = cv2.resize(img[0], dim, interpolation = cv2.INTER_CUBIC)
        b2 = cv2.resize(img[1], dim, interpolation = cv2.INTER_CUBIC)
        b3 = cv2.resize(img[2], dim, interpolation = cv2.INTER_CUBIC)
        
        img = np.dstack((b1,b2,b3))
        img = np.transpose(img,(2,0,1))
        return(img)
    
 
    def __getitem__(self,idx):
        
        current = self.df.iloc[idx]
        spot6_file = current["spot6_filenames"]
        sen2_files = current["sen2_filenames"]
        other_valid_acq = current["other_valid_acq"]    
        
        try:
            subfolder = str(current["subfolder"])
        except KeyError:
            pass
        
        if self.location == "colab":
            subfolder = "_sub/"+subfolder # 
        if self.location != "colab":
            subfolder = "" # keep subfolder empty to not alter input from dataframe


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
        spot6 = rasterio.open(self.folder_path+"y" + subfolder +"/" + spot6_file).read()
        

    
        """READ SEN2 SERIES"""
        # read first file
        sen2 = rasterio.open(self.folder_path+"x" + subfolder + "/" + sen2_files[0]).read()
        # match spot6 to first sen2 image
        sen2  = sen2/10000.0
        sen2_copy = sen2.copy() # safety copy to append if something goes wrong
        spot6 = spot6/255.0
        spot6 = self.histogram_matching(sen2,spot6)
        
        
        if self.sen2_amount>1:
            # read following sen2 and stack
            count=1
            for sen2_file in sen2_files[1:]:
                # read file as array
                sen2_following = rasterio.open(self.folder_path+"x"+subfolder+"/"+sen2_file).read()
                sen2_following  = sen2_following/10000.0
                # stack to previous images
                if sen2_following.shape==(3,75,75):
                    sen2 = np.concatenate([sen2, sen2_following])
                if sen2_following.shape!=(3,75,75):
                    print("warning, invalid sen2 shape. repeaeted first sen2 acquisition is concatenated instead")
                    sen2 = np.concatenate([sen2, sen2_copy])
                # break if all wanted files loaded
                count=count+1
                if count==self.sen2_amount:
                    break
            # if final count not yet reached, repeat last chip until enough are there
            while count<self.sen2_amount:
                print("warning: requested sen2 amount larger than available sen2 amount")
                sen2 = np.concatenate([sen2, sen2_copy]) # append if available sen2 amount < requested
                count=count+1
        

        # already stretched to stretch to 0..1
        sen2  = torch.from_numpy(sen2)
        spot6 = torch.from_numpy(spot6)
        sen2 = sen2.float()
        spot6 = spot6.float()


        # perform last sanity check, load random image if images arent expected shape
        while sen2.size()!= torch.Size([3* self.sen2_amount,75,75]) or spot6.size()!=torch.Size([3, 300, 300]):# or torch.sum(torch.where(sen2<=0,1,0)) > 1200*self.sen2_amount:
            print("Warning in DataLoader: wrong tensor size or more than 1.2k black pixels! Random image loaded instead")
            if self.transform=="spot6" or self.transform=="interpolate":
                break # dont do check if only doint spot 6 to spot 6
            sen2,spot6,coor = self.__getitem__(random.randint(0,self.__len__()-1))
        
        # return result
        return(sen2,spot6,(current["x"],current["y"]))
        

#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
import random
import geopandas
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

import warnings
import random
import time
import cv2


# In[2]:


# inputs
#spot6_mosaic = '/home/simon/CDE_UBS/thesis/data_collection/spot6/spot6_mosaic.tif'
spot6_path = "/content/drive/MyDrive/thesis/data/spot6/"
sen2_path = "/content/drive/MyDrive/thesis/data/sen2/"
#closest_dates_filepath = "/home/simon/CDE_UBS/thesis/data_loader/data/closest_dates.pkl"



# In[3]:


# Define torch dataset Class
class Dataset(Dataset):
    def __init__(self,spot6_mosaic,sen2_path,spot6_path,closest_dates_filepath,window_size=500,factor=(10/1.5),clip=True,temporal_images=1):
        
        # define filepaths
        self.temporal_images = temporal_images
        self.closest_dates_filepath = closest_dates_filepath
        self.spot6_mosaic = spot6_mosaic
        self.sen2_path = sen2_path
        self.spot6_path = spot6_path
        
        # define window size
        self.window_size = window_size
        self.factor = factor
        self.window_size_sen2 = int(window_size/self.factor)
        
        # define transformer
        self.transform_sen = transforms.Compose([
            transforms.Normalize(mean=[479.0, 537.0, 344.0], std=[430.0, 290.0, 229.0]) ])
        self.transform_spot = transforms.Compose([
            transforms.Normalize(mean=[78.0, 91.0, 62.0], std=[36.0, 28.0, 30.0]) ])
        
        
        # Get DF from create_data function!
        try:
            self.coordinates_closest_date_valid = pd.read_pickle("final_dataset.pkl")
            print("precalculated dataset file found, loading...")
        except FileNotFoundError:
            print("No precalculated dataset file found -> calling builder function!")
            from prepare_dataset import prepare_dataset
            self.coordinates_closest_date_valid = prepare_dataset(
                spot6_mosaic,sen2_path,spot6_path,closest_dates_filepath,
                window_size=500,factor=(10/1.5),clip=True,temporal_images=1)
        
        # delete invalid spot6 rows, reset index of df
        self.coordinates_closest_date_valid = self.coordinates_closest_date_valid[self.coordinates_closest_date_valid["spot6_validity"]==True]
        self.coordinates_closest_date_valid = self.coordinates_closest_date_valid[self.coordinates_closest_date_valid["index"]==162531]
        self.coordinates_closest_date_valid = self.coordinates_closest_date_valid.reset_index()
        
        
        # reset coordinates based on manipulated coordinates datasets
        tmp_coordinates = []
        for x,y in zip(self.coordinates_closest_date_valid["x"],self.coordinates_closest_date_valid["y"]):
            tmp_coordinates.append((x,y))
        self.coordinates = tmp_coordinates
        
        print("\nDataset Object successfully instanciated!")
 
    def __len__(self):
        """
        Returns length of data
        """
        return(len(self.coordinates_closest_date_valid))
    
 
    def __getitem__(self,idx):
        """
        Input:
            - id of item
        Output:
            - x and y images as np arrays
        """
        get_from_mosaic = False # set wether window should be taken from mosaic or individual spot6 files
        
        
        import time
        start_total = time.time()
        
        
        # extract coordinates of current request
        current_coor = self.coordinates[idx]
        current_coor_df = self.coordinates_closest_date_valid.loc[(self.coordinates_closest_date_valid["x"]==current_coor[0]) 
                                                                     & (self.coordinates_closest_date_valid["y"]==current_coor[1])]
        
        # load spot6 window by singular image
        if get_from_mosaic == False:
            start_spot6 = time.time()
            current_spot6_path = spot6_path + current_coor_df["name"][current_coor_df.index[0]]
            im_spot6 = Dataset.extract_spot6_window(current_spot6_path,coordinates=current_coor,window_size=self.window_size)
            end_spot6 = time.time()
        if get_from_mosaic == True:
            start_spot6 = time.time()
            im_spot6 = Dataset.extract_spot6_window(self.spot6_mosaic,coordinates=current_coor,window_size=self.window_size)
            end_spot6 = time.time()
        
        start_sen2 = time.time()
        
        # load sen2 window
        current_dict = current_coor_df["other_valid_acq"][current_coor_df.index[0]] # extract current dict
        
        val_sen2_paths = Dataset.get_valid_sen2paths(current_dict,sen2_path)
        
        
        # if valid return, straight to path extraction
        if type(val_sen2_paths)!=str:
            im_sen2 = Dataset.extract_sen2_window(val_sen2_paths,current_coor,self.window_size_sen2)
        
        # if invalid return, continue rand until valid patch found
        if type(val_sen2_paths)==str:
            warnings.warn("WARNING: Random image chosen to not break loader!")
            while type(val_sen2_paths)==str:
                current_coor = self.coordinates[random.randint(1,len(self.coordinates)-1)]
                current_coor_df = self.coordinates_closest_date_valid.loc[(self.coordinates_closest_date_valid["x"]==current_coor[0]) 
                                                                        & (self.coordinates_closest_date_valid["y"]==current_coor[1])]
                current_dict = current_coor_df["other_valid_acq"][current_coor_df.index[0]] # extract current dict
                val_sen2_paths = Dataset.get_valid_sen2paths(current_dict,sen2_path)
                im_sen2 = Dataset.extract_sen2_window(val_sen2_paths,current_coor,self.window_size_sen2)
            
        
        end_sen2 = time.time()
        
        
        end_total = time.time()
        #print("Total Time: ",end_total-start_total)
        #print("Spot6 Time: ",end_spot6-start_spot6)
        #print("Sen2 Time: ",end_sen2-start_sen2)
        # return extracted images
        
        
        """ CHECK FOR VALIDITY OF IMAGE SHAPES"""
        
        if im_spot6.shape != (3,self.window_size,self.window_size):
            warnings.warn("Warning in Dataloader: SPOT6 Window shape not valid at coordinates: "+str(current_coor)+" - Shape: "+str(im_spot6.shape))
        if im_sen2.shape != (3,self.window_size_sen2,self.window_size_sen2):
            warnings.warn("Warning in Dataloader: Sen-2 Window shape not valid at coordinates: "+str(current_coor)+" - Shape: "+str(im_sen2.shape))
            
        # resize for SR-CNN
        im_sen2 = Dataset.interpolate(im_sen2,size=self.window_size)
    
        # Perform Transform and change types
        im_sen2  = torch.from_numpy(im_sen2)
        im_spot6 = torch.from_numpy(im_spot6)
        
        im_sen2 = im_sen2.float()
        im_spot6 = im_spot6.float()
        
        # transform to 0..1 
        #im_sen2  = torch.div(im_sen2,10000.0)
        #im_spot6 = torch.div(im_spot6,255.0)
        
        # currently performing no normalization, only stretching to 0..1
        im_sen2  = self.transform_sen(im_sen2)
        im_spot6 = self.transform_spot(im_spot6)
        
        return(im_sen2,im_spot6)

    
    
    def interpolate(img,size=500):
        """
        Input:
            - Image
        Output:
            - Image upsampled to 500*500
        """
        dim = (size, size)
        b1 = cv2.resize(img[0], dim, interpolation = cv2.INTER_CUBIC)
        b2 = cv2.resize(img[1], dim, interpolation = cv2.INTER_CUBIC)
        b3 = cv2.resize(img[2], dim, interpolation = cv2.INTER_CUBIC)
        
        img = np.dstack((b1,b2,b3))
        img = np.transpose(img,(2,0,1))
 
        return(img)
    


    def extract_spot6_window(filepath,coordinates,window_size=500,show=False):
        """
        Inputs:
            - filepath of mosaic raster
            - point coordinates of window
            - window size in pixels
        Outputs:
            - window array from input mosaic at desired location
        
        """
        import rasterio
        import numpy as np

        # if coordinates == singular tuple of coordinates, wrap it in list
        if type(coordinates)!=list:
            coordinates = [coordinates]

        with rasterio.open(filepath) as dataset:
            # Loop through your list of coords
            for i, (lon, lat) in enumerate(coordinates):

                # Get pixel coordinates from map coordinates
                py, px = dataset.index(lon, lat)
                #print('Pixel Y, X coords: {}, {}'.format(py, px))

                # Build an NxN window (centered)
                window = rasterio.windows.Window(px - window_size//2, py - window_size//2, window_size, window_size)
                #print(window)

                # Read the data in the window
                # clip is a nbands * N * N numpy array
                clip = dataset.read(window=window)

                if show:
                    if clip.shape == (3, window_size, window_size):
                        image_standard_form = np.transpose(clip, (2, 1, 0))
                        plt.imshow(image_standard_form)
                        plt.show()
                    else:
                        print("Shape invalid - most likely edge window")

        return(clip)
 
    
    def extract_sen2_window(path_list,coordinates,window_size):
        """
        Inputs:
            - filepath of spot6 files
            - point coordinates of window as tuples
            - window size in pixels
        Outputs:
            - window array from input mosaic at desired location
        
        """
        import rasterio
        import numpy as np
        show=False # Show result?

        # extract coordinates
        lon,lat = coordinates[0],coordinates[1]
        # loop over list of acq.
        for file_path in path_list:
            # open file
            with rasterio.open(file_path) as dataset:
                # get pixel coordinates
                py,px = dataset.index(lon, lat)
                # build and read window
                window = rasterio.windows.Window(px - window_size//2, py - window_size//2, window_size, window_size)
                clip = dataset.read(window=window)

                # if wanted, show image
                if show:
                        if clip.shape == (3, window_size, window_size):
                            image_standard_form = np.transpose(clip, (2, 1, 0))
                            plt.imshow(image_standard_form)
                            plt.show()
                        else:
                            print("Shape invalid - most likely edge window")
        return(clip)
    
    def get_valid_sen2paths(acq_dict_sen2,path,num_images=1):
        dates = list(acq_dict_sen2.keys()) # get keys ergo closest times
        dates.sort() # sort so lowest can be accessed

        valid_files = []
        count=0
        count_true = 0
        for v,i in enumerate(dates): # iterate over closest
            if count_true==num_images: # stop while loop if number of required images is extracted
                #print("all extracted")
                break

            if acq_dict_sen2[i][2]==True:
                count_true=count_true+1
                filepath = acq_dict_sen2[i][1]
                valid_files.append(filepath)

            # protection for if more images requested than available
            if v==len(dates)-1:
                #warnings.warn("WARNING: More image aqc. dates requested than available. Recalculate full validity dataframe or request fewer sen2 images!")
                return("ERROR")

        for v,i in enumerate(valid_files):
            valid_files[v] = path + valid_files[v]
        return(valid_files)

    


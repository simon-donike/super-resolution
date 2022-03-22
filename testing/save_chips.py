#!/usr/bin/env python
# coding: utf-8

# In[1]:


# General Imports
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
from torch.autograd import Variable

import warnings
import random
import time
from tifffile import imsave

import skimage
from skimage.transform import rescale
import cv2

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import wandb


# In[2]:


# local imports
from dataloader_save_factor4 import Dataset


# In[3]:


# define paths
spot6_mosaic = '/home/simon/CDE_UBS/thesis/data_collection/spot6/spot6_mosaic.tif'
spot6_path = "/home/simon/CDE_UBS/thesis/data_collection/spot6/"
sen2_path = "/home/simon/CDE_UBS/thesis/data_collection/sen2/merged_reprojected/"
closest_dates_filepath = "/home/simon/CDE_UBS/thesis/data_loader/data/closest_dates.pkl"

# get dataset object
dataset = Dataset(spot6_mosaic,sen2_path,spot6_path,closest_dates_filepath,window_size=500,factor=(10/1.5))
loader = DataLoader(dataset,batch_size=1, shuffle=True, num_workers=1)
print("Loader Length: ",len(loader))


# In[ ]:


df = dataset.coordinates_closest_date_valid


# ## Save images to files

# In[ ]:


def extract_window(filepath,coordinates,window_size=500,show=False):
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


# In[ ]:


def extract_sen2_valid(dic):
    # find valid data
    valid_acq = []
    for i in dic:
        if dic[i][2] == True:
            valid_acq.append(i)
    return(valid_acq)
    


# In[ ]:


mosaic_path = "/home/simon/CDE_UBS/thesis/data_collection/spot6/spot6_mosaic.tif"
sen2_path   = "/home/simon/CDE_UBS/thesis/data_collection/sen2/merged_reprojected/"

spot6_export_path = "data_f4/y/"
sen2_export_path  = "data_f4/x/"


count = 0
spot6_filenames = []
sen2_filenames = []
sen2_number = []
sen2_tiles = []


for row in df.iterrows():
    """Extract Spoz6"""
    # extract spot6 image window
    spot6 = extract_window(mosaic_path,(row[1]["x"],row[1]["y"]))
    # set export filename
    spot6_name = str(row[1]["name"][:29])+"_"+str(row[1]["x"])+"_"+str(row[1]["y"])+".tif"
    # append to list
    spot6_filenames.append(spot6_name)
    # interpolate image
    spot6 = interpolate(spot6,300)
    # save image
    imsave(spot6_export_path+spot6_name, spot6)
    
    
    "Extract Sen2"
    # get dict valid inf
    dict_sen2 = row[1]["other_valid_acq"]
    sen2_valid_dates = extract_sen2_valid(dict_sen2)
    
    # iterate and save
    counter_sen2_files = 0
    sen2_file_names_ = []
    for i in sen2_valid_dates:
        # create export name
        sen2_export_name = dict_sen2[i][1][:61]+"_"+str(row[1]["x"])+"_"+str(row[1]["y"])+"_"+str(i)+str("days.tif")
        # get filepath+name of original image
        sen2_filename = sen2_path+dict_sen2[i][1]
        # get window
        sen2 = extract_window(sen2_filename,(row[1]["x"],row[1]["y"]),window_size=75)
        # increase counter to see how many sen2 images there are
        counter_sen2_files = counter_sen2_files+1
        # append filename to list to save
        sen2_file_names_.append(sen2_export_name)
        # SAVE
        imsave(sen2_export_path+sen2_export_name, sen2)
    sen2_tiles.append(sen2_export_name[35:41])
    
    
    sen2_filenames.append(sen2_file_names_)
    sen2_number.append(counter_sen2_files)
    
    count=count+1
    if count%100==0:
        print(count,"/",len(df),end="\r")
        
        
        
# append info to original dataframe
df["sen2_no"] = sen2_number
df["sen2_filenames"] = sen2_filenames
df["sen2_tile"] = sen2_tiles
df["spot6_filenames"] = spot6_filenames

df.to_pickle("df_saved_images.pkl")


# In[ ]:


df.to_csv("export")


# In[ ]:





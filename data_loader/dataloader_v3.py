#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import random
import geopandas

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


# In[3]:


# inputs
im_path_spot_2 = '/home/simon/CDE_UBS/thesis/data_collection/spot6/spot6_mosaic.tif'
closest_dates_filepath = "/home/simon/CDE_UBS/thesis/data_collection/sen2/closest_dates.pkl"


# In[4]:


# Define torch dataset Class
class Dataset(Dataset):
    def __init__(self,path,closest_dates_filepath,window_size=500,clip=False):
        # define raster filepath
        self.path = path
        # define window size
        self.window_size = window_size
        # create list of xy coordinates spaced according to window size over raster
        self.coordinates = Dataset.create_window_coordinates(self.path,window_size=self.window_size,clip=clip)
        # get closest sen2 acq. date for each datapoint and join with info on cell types
        self.coordinates_closest_date = Dataset.get_closest_date(self.coordinates,closest_dates_filepath)
        # test all sen2 coordinate windows for validity (warning, takes several hours!)
        self.coordinates_closest_date_valid = Dataset.create_sen2_validity_dataframe(self.coordinates_closest_date)
        
 
    def __len__(self):
        # returns length
        return(len(self.coordinates))
 
    def __getitem__(self,idx):
        # extract coordinates of current request
        current_coor = self.coordinates[idx]
        
        # load spot6 window
        im_spot6 = Dataset.extract_spot6_window(self.path,coordinates=current_coor,window_size=self.window_size)
        
        # load sen2 window
        """
        EXTRACT ONE CLOSEST DATE
        """
        # get according sen2 image
        # extract line from df that corresponds to this point
        #print(current_coor)
        #tmp1 = self.coordinates_closest_date[self.coordinates_closest_date["x"] == current_coor[0]]
        #print(tmp1)
        #tmp2 = tmp1[tmp1["y"] == current_coor[1]]
        #print(tmp2)
        #dates_dict = tmp2["other_acq"][0]
        #print(dates_dict)
        #print("value error")
        

        # return extracted images
        return(im_spot6)
    
    def extract_spot6_window(self,filepath,coordinates,window_size=500,show=False):
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


                """
                if save_file:
                    # Set Meta of new file
                    meta = dataset.meta
                    meta['width'], meta['height'] = N, N
                    meta['transform'] = rasterio.windows.transform(window, dataset.transform)

                    with rasterio.open(outfile.format(i), 'w', **meta) as dst:
                        dst.write(clip)
                """

        return(clip)
    
    def create_window_coordinates(filepath,window_size=500,clip=False):
        """
        Inputs:
            - fiepath: path of raster that is to be loaded by window
            - window_size: window will be pixel size NxN
            - clip: specify if every grid point should be sampled and dropped if value is invalid
        Outputs:
            - list of tuple coordinates of grid points (in CRS of input raster)
        Takes filepath, creates grid of coordinate points in wanted window size.
        (sampling of points bc mask reads whole into RAM)
        """

        # get bbox
        bbox = Dataset.get_spatial_extent(filepath)
        left = int(bbox[0])
        bottom = int(bbox[1])
        right = int(bbox[2])
        top = int(bbox[3])

        # iterate in N=window_size steps over image bounds, create grid
        coor = []
        for i in range(left,right,window_size):
            x = i
            for j in range(bottom,top,window_size):
                y = j
                coor.append((x,y))


        """
        PERFORM CLIP
        """
        if clip:
            import geopandas
            import pandas as pd
            import rasterio
            # load into gdf
            print("Performing clip of window corner points to valid raster values!\nloading points into gdf...")
            df = pd.DataFrame(coor,columns=["x","y"])
            gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y))

            print("verifying points on raster...")
            with rasterio.open(filepath) as src:
                gdf['value'] = [sum(x) for x in src.sample(coor)]

            print("dropping invalid points...")
            # drop invalid points and useless columns
            gdf = gdf.drop(gdf[gdf.value <= 0].index)
            # create new list of tuples to return
            coor = []
            for x_,y_ in zip(gdf["x"],gdf["y"]):
                coor.append((x_,y_))
            print("clipping done!\n")

        return(coor)

    def get_spatial_extent(filepath):
        """
        Takes filepath, returns bounding box
        """

        import rasterio
        with rasterio.open(filepath) as src:
            bbox = src.bounds
        return(bbox)
    
    def get_closest_date(coordinates,closest_dates_filepath):
        perform_train_test_split = True
        train_test_split_filepath = "train_test2.gpkg"
        
        
        import geopandas
        import fiona
        import pandas as pd
        
        print("Getting closest dates!")
        print("create closest dates gdf...")
        # load and transform closest dates dataframe
        df = pd.read_pickle(closest_dates_filepath)
        closest_dates = geopandas.GeoDataFrame(df, geometry=df.geom,crs=2154)
        del df

        print("create coordinates gdf...")
        # create coordinates gdf
        x,y = [],[]
        for i in coordinates:
            x.append(i[0])
            y.append(i[1])
        coordinates_df = pd.DataFrame()
        coordinates_df["x"] = x
        coordinates_df["y"] = y
        coordinates_df = geopandas.GeoDataFrame(coordinates_df, geometry=geopandas.points_from_xy(coordinates_df.x, coordinates_df.y),crs=2154)

        print("performing spatial join...")
        # spatial join for coordinates
        coordinates_joined_date = coordinates_df.sjoin(closest_dates, how="left")
        print("done\n")
        
        if perform_train_test_split:
            closest_date = coordinates_joined_date # rename file
            types = geopandas.read_file(train_test_split_filepath) # load GPKG file
            types = types.drop_duplicates(subset="name") # get rid of fuplicates
            types = types[["name","type"]] # keep only relevant columns
            coordinates_joined_date = closest_date.merge(types, on='name', how='inner', suffixes=('_1', '_2')) # join with df
            print("Train-Test split integrated into dataset!")
        return(coordinates_joined_date)
    

    def test_sen2_window(filepath,coordinates,window_size=100,show=False):
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

                if clip.shape == (3, window_size, window_size) and np.average(clip)>0.1:
                    validity = True

                    if show: # show image
                        image_standard_form = np.transpose(clip, (2, 1, 0))
                        #print(type(image_standard_form))
                        plt.imshow(image_standard_form)
                        plt.show()
                else:
                    validity = False

        return(validity)
    
    
    def create_sen2_validity_dataframe(df,sen2_folder_path="/home/simon/CDE_UBS/thesis/data_collection/sen2/merged_reprojected/"):
        """
        Inputs:
            - dataframe of coordinate points incl. Sen2 info
            - path to sen2 files
        Outputs:
            - DF w/ Sen3 dict appended with calidity information
            """
        print("\nChecking Sen2 validity for all windows & acquisitions - might take several hours")
        
        
        def test_sen2_window(filepath,coordinates,window_size=100,show=False): # inner function
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

                    if clip.shape == (3, window_size, window_size) and np.average(clip)>0.1:
                        validity = True

                        if show: # show image
                            image_standard_form = np.transpose(clip, (2, 1, 0))
                            #print(type(image_standard_form))
                            plt.imshow(image_standard_form)
                            plt.show()
                    else:
                        validity = False

            return(validity)

        
        
        # try to read precalculated file, if not recalculating
        try:
            df = pd.read_pickle("coordinates_validity_df.pkl")
            print("Precalculated File found - no recalculation")
            return(df)
        except FileNotFoundError:
            print("No precalculated file found, proceeding...")
        
        count=0
        ls_dict = []
        df_copy = df.copy(deep=True) # copy in order to not have append affect original file
        # iterate over rows in original df: dict of acq., x and y
        for dic,x,y in zip(df_copy["other_acq"],df_copy["x"],df_copy["y"]):
            dic_copy = dic
            #print(dic,x,y)
            dic_keys = dic.keys() # extract keys ergo acquisitions 
            # iterate over other acquisition dates
            for i in list(dic_keys):
                file = dic[i][1] # extract file name
                filepath = sen2_folder_path+file # save filepath
                temp_res = test_sen2_window(filepath,(x,y),window_size=50,show=False) # check validity
                if dic_copy[i][-1] != temp_res: # append only if information isn't present yet
                    dic_copy[i].append(temp_res)

            ls_dict.append(dic_copy) # append list w/ validity info to listwhich will be in DF

            count=count+1
            if count%100==0:
                perc = round(100 * float(count)/float(len(df)),2)
                print(str(perc),"%   ",end="\r")

        df["other_valid_acq"]=ls_dict
        df.to_pickle("coordinates_validity_df.pkl")
        return(df)




# In[ ]:


# Instanciate dataset object
dataset = Dataset(im_path_spot_2,closest_dates_filepath,window_size=500,clip=True)


# In[ ]:


# Instanciate dataloader object
loader = DataLoader(dataset,batch_size=1, shuffle=True, num_workers=1)
print("Loader Length: ",len(loader))


# In[ ]:





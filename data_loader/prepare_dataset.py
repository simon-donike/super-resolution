def prepare_dataset(spot6_mosaic,sen2_path,spot6_path,closest_dates_filepath,window_size=500, factor=(10/1.5),clip=True,temporal_images=1):
    import pandas as pd
    import matplotlib.pyplot as plt
    import random
    import geopandas
    import copy
    import numpy as np

    import warnings
    import random
    import time
    
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

                # Build an NxN window (top left corner), px - window_size//2, py - window_size//2
                window = rasterio.windows.Window(px, py, window_size, window_size)

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
    
    
    def check_spot6_validity(df,spot6_path,window_size=500):
        """
        Inputs:
            - dataframe with coordinates, file names for spot6
            - root path of spot6 images
            - window size for spot6
        Outputs:
            - list holding True/False values
            """
        print("\nChecking Spot6 Validity!")
        try:
            df = pd.read_pickle("coordinates_validity_spot6_df.pkl")
            print("Precalculated file found!")
        except FileNotFoundError:
            print("no precalculted file found, restarting calculation. This might take several hours...")
            ls = []
            counter = 0
            for x,y,file in zip(df["x"],df["y"],df["name"]):
                
                try:
                    tmp_image = extract_spot6_window(str(spot6_path+file),(x,y))

                    if tmp_image.shape == (3,window_size,window_size):
                        ls.append(True)
                    else:
                        ls.append(False)
                    counter=counter+1
                except:
                    ls.append(False)
                    warnings.warn("Exception in Spot6 Val. Check! For file: "+str(file))
                
                
                if counter%100==0:
                    perc = (100/len(df)) * counter
                    print("progress: ",round(perc,2),"%       ",end="\r")
            print("Done!\n")
            df["spot6_validity"] = ls
            df.to_pickle("coordinates_validity_spot6_df.pkl")
        return(df)
    
    
    def create_window_coordinates(filepath,window_size=500,clip=True):
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
        import numpy as np
        # get bbox
        bbox = get_spatial_extent(filepath)
        left = int(bbox[0])
        bottom = int(bbox[1])
        right = int(bbox[2])
        top = int(bbox[3])

        # iterate in N=window_size steps over image bounds, create grid
        coor = []
        for i in np.arange(left,right+1.5,window_size*1.5): # offset to also create "last" points, 1.5 for pix size in M
            x = i 
            for j in np.arange(bottom,top,window_size*1.5): # offset to also create "last" points, 1.5 for pix size in M
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
            print("clipping done!                        \n")

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
        """
        Inputs:
            - coordiantes df
            - filepath to closest dates vector
        Outputs:
            - joined DF of coordiantes with closest sen2 dates and paths
        """
        
        perform_train_test_split = False
        train_test_split_filepath = "train_test3.gpkg"
        
        
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

                if clip.shape == (3, window_size, window_size) and  np.median(clip)>0:      #(100 * float(np.count_nonzero(clip))/float(75*75*3))  > 0.10:
                    validity = True

                    if show: # show image
                        image_standard_form = np.transpose(clip, (2, 1, 0))
                        #print(type(image_standard_form))
                        plt.imshow(image_standard_form)
                        plt.show()
                else:
                    validity = False

        return(validity)
    
    
    def create_sen2_validity_dataframe(df,sen2_path,window_size_sen2):
        """
        Inputs:
            - dataframe of coordinate points incl. Sen2 info
            - path to sen2 files
        Outputs:
            - DF w/ Sen3 dict appended with calidity information
            """
        print("\nChecking Sen2 validity for all windows & acquisitions - might take several hours")
        
        
        def test_sen2_window(filepath,coordinates,window_size,show=False): # inner function
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
                    window = rasterio.windows.Window(px, py, window_size, window_size)
                    #print(window)

                    # Read the data in the window
                    # clip is a nbands * N * N numpy array
                    clip = dataset.read(window=window)

                    if clip.shape == (3, window_size, window_size) and np.median(clip)>0:#and (100 * float(np.count_nonzero(clip))/float(75*75*3))  > 0.10: # PERFORM CHECK IF MORE THAN 10PERC IS BLACK #and np.min(clip)>0.1:
                        validity = True

                        if show: # show image
                            image_standard_form = np.transpose(clip, (2, 1, 0))
                            #print(type(image_standard_form))
                            plt.imshow(image_standard_form)
                            plt.show()
                    else:
                        validity = False
            #print(validity)
            return(validity)
            # END INNER FUNCTION
        
        
        
        # try to read precalculated file, if not recalculating
        try:
            df = pd.read_pickle("coordinates_validity_sen2_df.pkl")
            print("Precalculated File found - no recalculation necessary!")
            return(df)
        except FileNotFoundError:
            print("No precalculated file found, calculating valid sen2 patches")
        
        # set amount of temporal checks for Sen2
        test_sen2_amount = 999
        print("Checking Sen2 temporal acquisition per point: ",test_sen2_amount)
        
        count=0
        ls_dict = []
        df_copy = df.copy(deep=True) # copy in order to not affect original file
        # iterate over rows in original df: dict of acq., x and y
        for dic,x,y in zip(df_copy["other_acq"],df_copy["x"],df_copy["y"]):
            dic_copy = copy.deepcopy(dic)
            dic_keys = dic.keys() # extract keys ergo acquisitions 
            dic_keys = list(dic_keys) # turn to list
            dic_keys.sort() # order list
            
            #print(x,y,dic_keys)
            
            # iterate over other acquisition date 
            counter_validity = 0 # counter that counts how many valid images were found yet
            for i in dic_keys:
                file = dic[i][1] # extract file name
                filepath = sen2_path+file # save filepath
                
                if counter_validity<=test_sen2_amount:
                    temp_res = test_sen2_window(filepath,(x,y),window_size_sen2,show=False) # check validity
                    
                    if temp_res==True:
                        counter_validity = counter_validity+1
                        dic_copy[i].append(temp_res)
                    if temp_res==False:
                        dic_copy[i].append(False)
                
                if counter_validity>test_sen2_amount: # if 3 valid reached, append False to further dates
                    dic_copy[i].append(False)
                
                #if dic_copy[i][-1] != temp_res: # append only if information isn't present yet
                #    dic_copy[i].append(temp_res)

            ls_dict.append(dic_copy) # append list w/ validity info to list which will be in DF

            count=count+1
            if count%100==0:
                perc = round(100 * float(count)/float(len(df)),2)
                print(str(perc),"%                   ",end="\r")

        df["other_valid_acq"]=ls_dict
        df.to_pickle("coordinates_validity_sen2_df.pkl")
        return(df)
    
    
    def extract_sen2_window(path_list,coordinates,window_size):
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
                # build and read window, adapt window: px - window_size//2,py - window_size//2
                window = rasterio.windows.Window(px, py, window_size, window_size)
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
    
    
    """
    CALLING FUNCTIONS
    """
    # try to read precalculated dataset
    try:
        coordinates_closest_date_valid = pd.read_pickle("final_dataset.pkl")
        print("Fully computed dataset found, no calculations necesary!")
    except FileNotFoundError:
        print("Full dataset not found, recalculating from scratch. This might take up to 12 hrs, depending on the availability of the Sen2/Spot6 validity files.\n\n")
        # define raster filepath
        temporal_images = temporal_images
        spot6_mosaic = spot6_mosaic
        sen2_path = sen2_path
        spot6_path = spot6_path

        # define window size
        window_size = window_size
        window_size_sen2 = int(window_size/factor)

        # create list of xy coordinates spaced according to window size over raster
        coordinates = create_window_coordinates(spot6_mosaic,window_size=window_size,clip=clip)

        # get closest sen2 acq. date for each datapoint and join with info on cell types
        coordinates_closest_date = get_closest_date(coordinates,closest_dates_filepath)

        # test all sen2 coordinate windows for validity (warning, takes several hours!)
        coordinates_closest_date_valid = create_sen2_validity_dataframe(coordinates_closest_date,sen2_path,window_size_sen2)
        # drop points where != train
        #coordinates_closest_date_valid = coordinates_closest_date_valid[coordinates_closest_date_valid["type"]=="train"]
        

        # check validity for spot6
        coordinates_closest_date_valid = check_spot6_validity(coordinates_closest_date_valid,spot6_path,window_size)

        # reset coordinates based on manipulated coordinates datasets, reset index
        coordinates_closest_date_valid = coordinates_closest_date_valid.reset_index()
        tmp_coordinates = []
        for x,y in zip(coordinates_closest_date_valid["x"],coordinates_closest_date_valid["y"]):
            tmp_coordinates.append((x,y))
        coordinates = tmp_coordinates
        coordinates_closest_date_valid.to_pickle("final_dataset.pkl")
        
    print("\nDataset successfully prepared!")
    
    return(coordinates_closest_date_valid)
    



# inputs
spot6_mosaic = '/home/simon/CDE_UBS/thesis/data_collection/spot6/spot6_mosaic.tif'
spot6_path = "/home/simon/CDE_UBS/thesis/data_collection/spot6/"
sen2_path = "/home/simon/CDE_UBS/thesis/data_collection/sen2/merged_reprojected/"
closest_dates_filepath = "/home/simon/CDE_UBS/thesis/data_loader/data/closest_dates.pkl"


# In[ ]:nan


__ = prepare_dataset(spot6_mosaic,sen2_path,spot6_path,closest_dates_filepath,window_size=500,factor=(10/1.5),clip=True,temporal_images=1)




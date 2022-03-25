import pandas as pd
import os
import random

df = pd.read_pickle("df_saved_images.pkl")

path_x = "/home/simon/CDE_UBS/thesis/testing/data_f4/x/"
path_y = "/home/simon/CDE_UBS/thesis/testing/data_f4/y/"

new_path_x = "/home/simon/CDE_UBS/thesis/testing/data_f4/x_sub/"
new_path_y = "/home/simon/CDE_UBS/thesis/testing/data_f4/y_sub/"

l = []
counter = 0
for sen,spot in zip(df["sen2_filenames"],df["spot6_filenames"]):
    new_path = random.randint(1,99)
    
    # sen2
    for i in sen:
        os.system("cp "+path_x+i+" "+new_path_x+str(new_path)+"/"+i)
        
    # spot 6
    os.system("cp "+path_y+spot+" "+new_path_y+str(new_path)+"/"+i)
    
    l.append(new_path)
    
    
    counter = counter+1
    if counter%10==0:
        print(counter,"/",len(df),"           ",end="\r")
df["subfolder"]=l
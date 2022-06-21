#!/usr/bin/env python
# coding: utf-8

# SENTINEL2A_20180403-113452-260_L2A_T30UUU_C_V2-2_FRE_B3.tif
"""
gdal_merge.py -o merged/out.tif  SENTINEL2A_20180802-105938-888_L2A_T30UXU_C_V2-2/SENTINEL2A_20180802-105938-888_L2A_T30UXU_C_V2-2_FRE_B2.tif SENTINEL2A_20180802-105938-888_L2A_T30UXU_C_V2-2/SENTINEL2A_20180802-105938-888_L2A_T30UXU_C_V2-2_FRE_B3.tif SENTINEL2A_20180802-105938-888_L2A_T30UXU_C_V2-2/SENTINEL2A_20180802-105938-888_L2A_T30UXU_C_V2-2_FRE_B4.tif

gdal_merge.py -separate  -o new_rgb.tif -co PHOTOMETRIC=MINISBLACK C:\input_r.tif C:\input_g.tif C:\input_b.tif

"""
import sys,os
import time


out_file_path = "/share/projects/erasmus/sesure/sen2/data/merged/"


print("start finding B-1-2-3")
start_time = time.time()
# define input path
root = "/share/projects/erasmus/sesure/sen2/"
#path = os.path.join(root, "targetdirectory")
path = root




create_new = True
if create_new:
    # get list of all tif files in directory & subdirectory
    tifs = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[-10:] == "FRE_B2.tif":
                tifs.append(os.path.join(path, name))
    print("No. of Band-1-2-3 files: ",len(tifs))




    # create list of commands from files
    commands = []
    for i in tifs:
        location = i[:i.rfind("/")+1] # path of file
        
        # RGB Channel File Names
        r = i[i.rfind('/')+1:] # file name
        g = r[:-5] + "3.tif"
        b = r[:-5] + "4.tif"
        
        out_file_name = out_file_path + r[:-6] + "RGB.tif"
        command = "gdal_merge.py -separate  -o " + out_file_name +" "+location+b+" "+location+g+" "+location+r
        commands.append(command)
    print("No. of commands: ",len(commands),"\nExample: ",commands[0])

    # write commands to file
    with open('commands_merge.txt', 'w') as f:
        for item in commands:
            f.write("%s\n" % item)
            



# In[ ]:



print("FINISHED SETUP")
time.sleep(5)   
print("\nSTARTING MERGING\n")
counter = 0
no_tifs = len(tifs)
for i in commands:
    counter = counter+1
    os.system(i)
    if counter %1==0:
        with open("log.txt", "a") as myfile:
            percentage = round(100 * float(counter)/float(no_tifs),2)
            temp_time = time.time()-start_time
            print_str = "image no: "+str(counter)+" after "+str(round(temp_time/60,2))+" min or "+str(round(temp_time/3600,2)) + " hrs. - "+str(percentage)+"%\n"
            myfile.write(print_str)
    
print("Done!")






# 2154


import sys,os
import time
print("start finding tiffs")
start_time = time.time()
# define input path
root = "/share/projects/erasmus/sesure/sen2/"
out_dir = "/share/projects/erasmus/sesure/sen2/data/merged_reprojected/"
path = os.path.join(root, "targetdirectory")

create_new = True
if create_new:
    # get list of all tif files in directory & subdirectory
    tifs = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[-7:] == "RGB.tif":
                tifs.append(os.path.join(path, name))
    print("No. of .tif files: ",len(tifs))



    # create list of commands from files
    commands = []
    for i in tifs:
        location = i[:i.rfind("/")+1] # path of file
        in_file_name = i[i.rfind('/')+1:]
        out_file_name = out_dir + in_file_name[:in_file_name.rfind(".")]+"_2154.tif"
        command = "gdalwarp -t_srs EPSG:2154 " + location+in_file_name + " "  + out_file_name
        commands.append(command)
    print("No. of commands: ",len(commands),"\nExample: ",commands[0])

    # write commands to file
    with open('commands_reproject.txt', 'w') as f:
        for item in commands:
            f.write("%s\n" % item)
            


print("FINISHED SETUP")
time.sleep(5)   
print("\nSTARTING REPROJECT\n")
counter = 0
no_tifs = len(tifs)
for i in commands:
    counter = counter+1
    os.system(i)
    if counter %1==0:
        with open("log_reproject.txt", "a") as myfile:
            percentage = round(100 * float(counter)/float(no_tifs),2)
            temp_time = time.time()-start_time
            print_str = "image no: "+str(counter)+" after "+str(round(temp_time/60,2))+" min or "+str(round(temp_time/3600,2)) + " hrs. - "+str(percentage)+"%\n"
            myfile.write(print_str)
print("Done!")

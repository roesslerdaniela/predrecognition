import csv
import os

'''
Last changed: 2021-02-20
README

Use this to prepare an initial csv with DLC csv filenames, MOV filenames, and a scaling factor.

Note: Spider position is referred to as "midpoint". 

How:
1. Place this in folder with DLC csvs, MOV files, scaling factor csv, and where you want to use dlcvideo2spiderdata.py
2. Replace the path variable with the path to the folder with all of the above
3. Replace the scaling_csv variable with the name of the csv scaling file of choice
4. Run it
5. Run dlvcideo2spiderdata.py in the same folder
'''
#replace this path with the folder path that contains the DLC outputs, MOV files, and sclaing factor csv
path = "C:/Users/ShambleLab/Documents/condaDLCenv/2020_04_20_salticus/videos/"
scaling_csv = "PredExp_px_to_mm_conversionData.csv" #replace this with the file name of the scaling factor csv

scale_name = [] #list to hold the uncropped video name
scale_num = [] #list to the corresponding scale factors

#open the mm/px scaling csv and extract the corresponding uncropped video names and scaling factors
with open(path+scaling_csv, newline='') as csvfile:
    readcsv = csv.reader(csvfile, delimiter=',')
    next(readcsv) #skip the first title row
    for row in readcsv:
        scale_name.append(row[0])
        scale_num.append(row[10])

csv_list_of_files = open(path+"dlc_video_names.csv",'w', newline="") #create the path to dlc_video_names.csv
csv_writer = csv.writer(csv_list_of_files) #create a csv writer function

directory = os.fsencode(path) #define the directory that you want to iterate through
video_list = [] #list to hold cropped video names
dlc_list = [] #list to hold dlc csv names

#iterate through the files in the folder and extract all the cropped video names
for file in os.listdir(directory):
    filename = os.fsdecode(file) #grab the file name
    if filename.endswith(".MOV"):
        video_list.append(str(filename))

#iterate through the files in the folder and extract all the dlc csv names
for file in os.listdir(directory):
    filename = str(os.fsdecode(file))
    if filename.endswith(".csv") and ("DLC" in filename) and ("_sightline" not in filename) and ("_vectors" not in filename):
        dlc_list.append(filename)

video_list.sort() #sort the cropped video names
dlc_list.sort() #sort the dlc csv names

write_array = [0, 0, 0] #make a temporary list to write rows to the dlc_video_names csv

#iterate through the cropped video names and write the cropped video name, its corresponding dlc csv name, and its corresponding mm/px to dlc_video_names csv
for i in range(len(video_list)):
    write_array[0] = dlc_list[i] #write dlc csv name
    write_array[1] = video_list[i] #write cropped video name
    write_array[2] = 0 #placeholder
    noncrop_name = video_list[i].split("_cropped_")[0] + ".MOV" #extract the uncropped video name
    if noncrop_name in scale_name: #search for the uncropped video name in the uncropped video list
        scale_ind = scale_name.index(noncrop_name) #find the index
        write_array[2] = scale_num[scale_ind] #find the corresponding mm/px factor
    csv_writer.writerow(write_array) #write to the dlc_video_names csv
csv_list_of_files.close() #close your csv file

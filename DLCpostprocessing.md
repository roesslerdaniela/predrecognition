DLC + Post Processing Code Procedure

1. Run the DLC project on all videos and output a csv of all labeled points for each video
2. Create a experiments data csv with trial information for each video (spider characteristics, trial start time (where the video was front cropped), trial end time, etc)
3. Create the scaling facor csv with a list of videos and their corresponding mm/px
4. Place the DLC csvs, MOV files, scaling factor csv, experiments data csv, and all the post processing python code inthe same folder
5. Run prep_dlcvideo2spiderdata_files.py
6. Run dlcvideo2spiderdata.py
7. Run sightline_processing.py

Notes:  
- Video names were altered with "_cropped_SECONDS-MILLISECONDS" when front cropped before being analyzed by DLC, but the original video names were used in scaling csv and experiment data csv. As a result, there are numerous points in the post processing code that make sure that corresponding trial data from one csv lined up properly with the trial data from another csv. These code segments can be condensed if the video names stay consistent. 
- Scaling factor csv is used in prep_dlcvideo2spiderdata_files.py. If your scaling factor csv is different, only the column numbers the code is pulling from need to be replaced (or if the scaling factor is all the same across videos, the variable itself can be reassigned to a single number).
- The experiment data csv has a whole list of experimental characteristics for each trial. The "trial end" was determined by video observation, and this endtime variable was used in determining the end of the trial in sightline_processing.py before calculating longest freeze/retreat periods. 

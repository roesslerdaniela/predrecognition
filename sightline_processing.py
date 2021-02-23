import os
import csv
import numpy as np
import scipy.signal
import math

'''
Last changed: 2021-02-22
README

Only run this code if you've already run dlcvideo2spiderdata_files.py!

Note: "target" and "predator" both refer to the object on the opposite platform

Use this code to:
1. Generate four csvs of freeze and retreat characteristics, one for each "looking" condition - angle offset, 10deg cone, 28deg cone, and 56deg cone

How:
1. Make sure this code is in the same folder as ALL the MOVs, csvs, and post processing code
2. Replace the path variable with the path to the folder with everything
3. Replace the results_csv with the name of the csv with overall experiment characteristics
4. Run sightline_processing.py
'''

#function to find the start and end indices of sections of 1's in boolean arrays
def find_true_bookends(bool_np_array, frame_index_array):
    # find the indices of the start and end of the sections of 1's, sorted
    one_to_zero = np.where(bool_np_array[:-1] > bool_np_array[1:])[0]
    zero_to_one = np.where(bool_np_array[:-1] < bool_np_array[1:])[
                      0] + 1  # shift by one to make sure you get a 1 index, not a 0 index

    one_shifts = np.sort(np.append(one_to_zero, zero_to_one))

    if bool_np_array[0] > 0:  # check if the front segement is 1's and add a bookend
        one_shifts = np.insert(one_shifts, 0, 0)
    if len(one_shifts) % 2 > 0:  # check if the last segment is 1's and add a bookend
        one_shifts = np.append(one_shifts, len(overlapping_array) - 1)

    # find the corresponding frame numbers for the boolean switches
    frame_bookends = np.zeros(len(one_shifts))
    for i in range(len(one_shifts)):
        frame_bookends[i] = frame_index_array[one_shifts[i]]

    return frame_bookends

# replace this path with the folder path that contains the DLC outputs and MOV files
path = "C:/Users/ShambleLab/Documents/condaDLCenv/2020_04_20_salticus/Sightlines/"
directory = os.fsencode(path)  # define the directory that you want to iterate through
def_retreat_len = 120 #number of frames after longest freeze that is considered the "retreat" region
sg_poly_val = 3 #Savitzky-Golay filter poly value
sg_velo_window_val = 25 #Savitzky-Golay filter window for velocity
sg_window_val = 17 #general Savitzky-Golar filer window
angle_offset_rad_val = 0.38 #angle offset boundary for "looking"

# open Daniela's results excel file to grab experiment data
results_csv = 'RESULTS_ExperimentsGermany_2020-07-21 - RESULTS_ExperimentsGermany_2020-07-21.csv'
video_name = []
experiment = []
subject = []
sex = []
day = []
condition = []
trial = []
date = []
test = []
endtime = []

with open(path + results_csv, newline='') as csvfile:
    readcsv = csv.reader(csvfile, delimiter=',')
    next(readcsv)  # skip the first title row
    for row in readcsv:
        video_name.append(row[23])
        experiment.append(row[0])
        subject.append(row[1])
        sex.append(row[2])
        day.append(row[3])
        condition.append(row[4])
        trial.append(row[5])
        date.append(row[6])
        test.append(row[7])
        endtime.append(row[9])
csvfile.close()

# all the dictionaries
#freeze_dic = {}
#data_dlc = {}
file_list = []
#calculated_angles_dict = {}
data_list = ['sex', 'day', 'date', 'experiment', 'spiderID', 'trial', 'condition', 'test', 'video name',
             'targetposx', 'targetposty', 'cropped video name', 'conversion factor', 'freeze', 'freezeposx',
             'freezeposy',
             'freezedist', 'freezewindow', 'freezestartframe', 'freezeangleavg',
             'retreatendposx', 'retreatendposy', 'retreatendangle', 'retreatenddist', 'retreatwindow',
             'retreatdistchange', 'retreatposchange', 'retreatavgspeed', 'retreatavgspeedangle',
             'retreatnumframes', 'targetpass', 'jump', 'retreatlookingratio', 'totallookingratio']

#list of "looking" conditions
looking_conditions = ['angle_offset', '10degcone', '28degcone', '56degcone']

#set up the list of results for each definition of "looking"
angle_offset_data_list = [] #looking is within angle_offset_rad_value
cone_10_data_list = [] #looking is having the predator hull intersect a 10deg view cone
cone_28_data_list = [] #looking is having the predator hull intersect a 28deg view cone
cone_56_data_list = [] #looking is having the predator hull intersect a 56deg view cone

#copy the column header into each list
angle_offset_data_list.append(data_list[:])
cone_10_data_list.append(data_list[:])
cone_28_data_list.append(data_list[:])
cone_56_data_list.append(data_list[:])

for file in os.listdir(directory):
    filename = os.fsdecode(file)  # grab the file name
    if "sightline" in filename:
        file_list.append(filename)
        sightline_file = np.loadtxt(path + filename, delimiter=",")
        sightline_file = sightline_file.astype(float)

        #grab the test's characteristics
        uncropped_vid_name = filename.split("_cropped_")[0] + ".MOV" #video name
        results_ind = video_name.index(uncropped_vid_name) #find where the corresponding row in Daniela's results csv
        second_half = filename.split("_cropped_")[1]
        cropstart = second_half.split("DLC")[0]
        cropstart_seconds = float(cropstart.split("-")[0] + "." + cropstart.split("-")[1]) #find the time where the video was cropped at the beginning
        cropend_seconds = float(endtime[results_ind]) #find where observation says the experiment finished
        frame_cutoff = (cropend_seconds - cropstart_seconds) * 59.94 #find the specific end of experiment frame in the cropped video
        frame_cutoff_index = np.abs(sightline_file[:, 0] - frame_cutoff).argmin() #find the index of this final experimental frame

        sightline_file = sightline_file[:frame_cutoff_index + 1] #crop the sightline data so that it ends where observation says the experiment ended

        #filter the spider velocity, angle offset, x position, y position, and distance from the predator
        smooth_velocity = scipy.signal.savgol_filter(sightline_file[:, 6], sg_velo_window_val, sg_poly_val)
        smooth_angle = scipy.signal.savgol_filter(sightline_file[:, 3], sg_window_val, sg_poly_val)
        smooth_x = scipy.signal.savgol_filter(sightline_file[:, 1], sg_window_val, sg_poly_val)
        smooth_y = scipy.signal.savgol_filter(sightline_file[:, 2], sg_window_val, sg_poly_val)
        smooth_distance = scipy.signal.savgol_filter(sightline_file[:, 4], sg_window_val, sg_poly_val)

        '''
        calc_angle_offset_rad_val = []
        #Calculating angle_offset_rad_val by using predator bounding box height
        for i in range(len(sightline_file[:,0])):
            midpoint_top = np.array([sightline_file[:,13][i]-sightline_file[:,1][i], sightline_file[:,14][i]-sightline_file[:,2][i]])
            midpoint_bottom = np.array([sightline_file[:,17][i]-sightline_file[:,1][i], sightline_file[:,18][i]-sightline_file[:,2][i]])
            bounding_angle = np.arccos(np.dot(midpoint_top, midpoint_bottom) / (
                    np.linalg.norm(midpoint_top) * np.linalg.norm(midpoint_bottom)))
            calc_angle_offset_rad_val.append(bounding_angle/2.0)
        calculated_angles_dict[filename] = calc_angle_offset_rad_val
        '''

        #iterate through each sightline file four times, one for each "looking" condition
        for cycle in looking_conditions:
            freeze_bookends = [] #list of freeze bookend frames
            freeze_lengths = [] #list of freeze lengths in frames

            # extract the velocity values that are <1px/s from the sightline file
            velocity_px_binary = np.zeros(len(sightline_file[:, 0]))
            for i in range(len(sightline_file[:, 6])):
                if ((smooth_velocity[i]) < 1.0):
                    stopped = 1
                else:
                    stopped = 0
                velocity_px_binary[i] = stopped

            # make a binary array of all the frames where the spider is looking at the predator
            # this will depend on the looking condition
            angle_offset_binary = np.zeros(len(sightline_file[:, 0]))
            if cycle == 'angle_offset':
                # extract the angle offset values that are <0.38radians from the sightline file
                for i in range(len(sightline_file[:, 3])):
                    if ((smooth_angle[i]) < angle_offset_rad_val):
                        looking = 1
                    else:
                        looking = 0
                    angle_offset_binary[i] = looking

            elif cycle == '10degcone':
                angle_offset_binary = sightline_file[:, 19]

            elif cycle == '28degcone':
                angle_offset_binary = sightline_file[:, 20]

            elif cycle == '56degcone':
                angle_offset_binary = sightline_file[:, 21]

            # calculate relative target velocity and extract when that velocity is in the negative
            predator_distance = sightline_file[:, 4]
            relative_target_velocity = np.zeros(len(predator_distance))
            retreat_binary = np.zeros(len(predator_distance))
            for i in range(1, len(predator_distance)):
                relative_target_velocity[i] = (predator_distance[i] - predator_distance[i - 1]) / (
                            sightline_file[:, 0][i] - sightline_file[:, 0][i - 1])
            smooth = scipy.signal.savgol_filter(relative_target_velocity, 41, 3) #smooth the relative target velocity
            for i in range(len(relative_target_velocity)): #make a binary array for when the spider is moving away from the predator
                if smooth[i] > 0:
                    retreat_binary[i] = 1

            # make sure that both boolean lists are in numpy arrays
            velocity_binary_array = np.array(velocity_px_binary)
            angle_binary_array = np.array(angle_offset_binary)
            retreat_binary_array = np.array(retreat_binary)

            #get the boolean array where both the spider is looking at the predator and velocity<1px/s == 1
            overlapping_array = velocity_binary_array * angle_binary_array

            #get the boolean array where both the spider is retreating and looking at the predator
            retreat_overlapping_array = retreat_binary_array * angle_binary_array

            overlapping_frame_bookends = find_true_bookends(overlapping_array,
                                                            sightline_file[:, 0])  # find freezing indices
            paused_frame_bookends = find_true_bookends(velocity_binary_array,
                                                       sightline_file[:, 0])  # find pausing indices
            away_frame_bookends = find_true_bookends(retreat_binary_array, sightline_file[:,
                                                                           0])  # find indices where spider is moving away from the model
            retreat_frame_bookends = find_true_bookends(retreat_overlapping_array,
                                                        sightline_file[:, 0])  # find retreating indices

            freezes = np.array([])  # make an array to store freezing values
            if len(overlapping_frame_bookends) > 0:
                overlapping_frame_bookends_pairs = np.split(overlapping_frame_bookends,
                                                            len(overlapping_frame_bookends) / 2)
                for elements in overlapping_frame_bookends_pairs:
                    if (elements[1] - elements[0]) > 0.0: #check for actual freezes
                        freezes = np.append(freezes, elements)
                        freeze_bookends.append([elements])  #add the frame bookends to the list
                        freeze_lengths.append(elements[1] - elements[0]) #add the corresponding freeze lengths to the list

            '''
            pauses = np.array([])
            if len(paused_frame_bookends) > 0:
                paused_frame_bookends_pairs = np.split(paused_frame_bookends, len(paused_frame_bookends)/2)
                for elements in paused_frame_bookends_pairs:
                    if (elements[1] - elements[0]) > 60.0:
                        pauses = np.append(pauses, elements)

            away = np.array([])
            if len(away_frame_bookends) > 0:
                away_frame_bookends_pairs = np.split(away_frame_bookends, len(away_frame_bookends) / 2)
                for elements in away_frame_bookends_pairs:
                    if (elements[1] - elements[0]) > 50.0:
                        away = np.append(away, elements)

            retreats = np.array([])
            if len(retreat_frame_bookends) > 0:
                retreat_frame_bookends_pairs = np.split(retreat_frame_bookends, len(retreat_frame_bookends)/2)
                for elements in retreat_frame_bookends_pairs:
                    if (elements[1]-elements[0]) > 50.0:
                        retreats = np.append(retreats, elements)
            '''

            # check if the spider ever passes
            # this is defined as the spider x < predator center x AND being located within 20px of the predator center
            passed_binary = np.zeros(len(smooth_x))
            close_binary = np.zeros(len(sightline_file[:, 4]))
            for i in range(len(smooth_x)):
                if smooth_x[i] < sightline_file[:, 9][0]:
                    passed = 1
                else:
                    passed = 0
                passed_binary[i] = passed
            for i in range(len(sightline_file[:, 5])):
                if sightline_file[:, 5][i] < 20:
                    close = 1
                else:
                    close = 0
                close_binary[i] = close
            pass_model = passed_binary * close_binary

            # check if the spider jumps across the gap
            jump_binary = np.zeros(len(smooth_x))
            for i in range(len(smooth_x)):
                if smooth_x[i] < 960.0:
                    jump = 1
                else:
                    jump = 0
                jump_binary[i] = jump

            #load current/placeholder values for this sightline file's datalist
            data_list[0] = sex[results_ind]  # sex
            data_list[1] = day[results_ind]  # day
            data_list[2] = date[results_ind]  # date
            data_list[3] = experiment[results_ind]  # experiment
            data_list[4] = subject[results_ind]  # spiderID
            data_list[5] = trial[results_ind]  # trialnum
            data_list[6] = condition[results_ind]  # condition
            data_list[7] = test[results_ind]  # test
            data_list[8] = uncropped_vid_name  # vidname

            data_list[9] = sightline_file[:, 9][0]  # targetposx
            data_list[10] = sightline_file[:, 10][0]  # targetposy
            data_list[11] = filename.split("DLC")[0] + ".MOV"  # cropvidname
            data_list[12] = sightline_file[:, 8][0]  # conversionfactor

            data_list[13] = len(freeze_lengths) > 0  # freeze Y/N?
            data_list[14] = 0  # avgfreezeposx
            data_list[15] = 0  # avgfreezeposy
            data_list[16] = 0  # avgfreezedist
            data_list[17] = 0  # freezeduration
            data_list[18] = 0  # freezestartind
            data_list[19] = 0  # avgfreezeang

            data_list[20] = 0  # retreatendposx
            data_list[21] = 0  # retreatendposy
            data_list[22] = 0  # retreatendang
            data_list[23] = 0  # retreatenddist
            data_list[24] = 0  # retreatduration
            data_list[25] = 0  # retreatdistchange
            data_list[26] = 0  # retreatposchange
            data_list[27] = 0  # retreatspeedavg
            data_list[28] = 0  # retreatspeedavgang
            data_list[29] = 0  # numgoodframes
            data_list[30] = sum(pass_model) > 0  # targetpass
            data_list[31] = sum(jump_binary) > 0  # jump
            data_list[32] = 0  # % of retreat looking at predator
            data_list[33] = np.sum(angle_offset_binary) / len(angle_offset_binary)  # % of frames that the spider is looking at the target

            #if the spider ever freezes, find the longest freeze and calculate freezing and retreat characteristics based off that freeze
            sightline_frame_list = sightline_file[:, 0]
            if len(freeze_lengths) > 0:
                #find freeze and retreat frame indices and durations
                max_freeze = max(freeze_lengths)
                max_freeze_index = freeze_lengths.index(max_freeze)
                start_frame, end_frame = freeze_bookends[max_freeze_index][0]
                freeze_duration = end_frame - start_frame
                retreat_start_index = np.abs(sightline_frame_list - (end_frame + 1)).argmin()
                retreat_end_index = np.abs(sightline_frame_list - (end_frame + def_retreat_len + 1)).argmin()
                freeze_start_index = int(np.where(sightline_frame_list == start_frame)[0])
                freeze_end_index = int(np.where(sightline_frame_list == end_frame)[0])
                retreat_duration = sightline_frame_list[retreat_end_index] - sightline_frame_list[retreat_start_index]

                average_angle = sum(sightline_file[:, 3][freeze_start_index:freeze_end_index + 1]) / len(
                    sightline_file[:, 1][freeze_start_index:freeze_end_index + 1]) #calculate average angle offset over all existing freezing frames
                average_distance = sum(sightline_file[:, 4][freeze_start_index:freeze_end_index + 1]) / len(
                    sightline_file[:, 1][freeze_start_index:freeze_end_index + 1]) #calculate average distance from predator over all existing freezing frames
                average_pos = [sum(smooth_x[freeze_start_index:freeze_end_index + 1]) / len(
                    sightline_file[:, 1][freeze_start_index:freeze_end_index + 1]),
                               sum(smooth_y[freeze_start_index:freeze_end_index + 1]) / len(
                                   sightline_file[:, 1][freeze_start_index:freeze_end_index + 1])] #calculate average x,y freezing position over all existing freezing frames

                pos_retreat_end = [smooth_x[retreat_end_index], smooth_y[retreat_end_index]] #find spider position at the end of the retreat
                dist_retreat_end = sightline_file[:, 4][retreat_end_index] #find the distance from predator at the end of the retreat
                distance_diff = (sightline_file[:, 4][retreat_end_index] - sightline_file[:, 4][retreat_start_index]) #find the change in distance from the predator over the retreat
                ang_off_end = sightline_file[:, 3][retreat_end_index] #find the angle offset at the end of the retreat
                pos_diff = math.sqrt(((smooth_x[retreat_end_index] - smooth_x[retreat_start_index]) ** 2) + (
                            (smooth_y[retreat_end_index] - smooth_y[retreat_start_index]) ** 2)) #find the difference in spider position over the retreat
                average_speed_retreat = pos_diff / (sightline_frame_list[retreat_end_index] - sightline_frame_list[
                    retreat_start_index])  #find the average retreat speed over all possible retreat frames, invalid value if retreat start and end are the same
                start_to_end = np.array([smooth_x[retreat_end_index] - smooth_x[retreat_start_index],
                                         smooth_y[retreat_end_index] - smooth_y[retreat_start_index]]) #vector from the retreat start position to the retreat end position
                start_to_hz = np.array([-1, 0]) #hz vector
                average_speed_retreat_angle = np.arccos(np.dot(start_to_hz, start_to_end) / (
                            np.linalg.norm(start_to_hz) * np.linalg.norm(
                        start_to_end))) #average retreat angle as measured from the horizontal, invalid value if retreat and end are the same
                num_retreat_frames = len(sightline_file[:, 1][retreat_start_index:retreat_end_index + 1]) #total number of retreat frames

                #calculate how many of the retreat frames the spider looks at the predator
                retreatlookcount = 0
                if cycle == 'angle_offset':
                    for i in range(retreat_start_index, retreat_end_index + 1):
                        if sightline_file[:, 3][i] < angle_offset_rad_val:
                            retreatlookcount += 1
                elif cycle == '10degcone':
                    retreatlookcount = sum(sightline_file[:, 19][retreat_start_index:retreat_end_index+1])
                elif cycle == '28degcone':
                    retreatlookcount = sum(sightline_file[:, 20][retreat_start_index:retreat_end_index + 1])
                elif cycle == '56degcone':
                    retreatlookcount = sum(sightline_file[:, 21][retreat_start_index:retreat_end_index + 1])

                data_list[14] = average_pos[0]  # avgfreezeposx
                data_list[15] = average_pos[1]  # avgfreezeposy
                data_list[16] = average_distance  # avgfreezedist
                data_list[17] = freeze_duration  # freezeduration
                data_list[18] = start_frame  # freezestartind
                data_list[19] = average_angle  # avgfreezeang

                data_list[20] = pos_retreat_end[0]  # retreatendposx
                data_list[21] = pos_retreat_end[1]  # retreatendposy
                data_list[22] = ang_off_end  # retreatendang
                data_list[23] = dist_retreat_end  # retreatenddist
                data_list[24] = retreat_duration  # retreatduration
                data_list[25] = distance_diff  # retreatdistchange
                data_list[26] = pos_diff  # retreatposchange
                data_list[27] = average_speed_retreat  # retreatspeedavg
                data_list[28] = average_speed_retreat_angle  # retreatspeedavgang
                data_list[29] = num_retreat_frames  # numgoodframes
                data_list[32] = retreatlookcount / num_retreat_frames  # %retreat looking at predator

            #write freeze/retreat characteristics to their respective datalists
            if cycle == 'angle_offset':
                angle_offset_data_list.append(data_list[:])
            elif cycle == '10degcone':
                cone_10_data_list.append(data_list[:])
            elif cycle == '28degcone':
                cone_28_data_list.append(data_list[:])
            elif cycle == '56degcone':
                cone_56_data_list.append(data_list[:])
            print(cycle)

        print(filename)

#write each datalist to its own csv file
csv_angle_offset_data_list = open(path + "2020_12_11_datablock_angleoffset.csv", 'w',newline="")  # create a csv file to collect spider sight angles, speed, distance, etc
csv_angle_offset_writer = csv.writer(csv_angle_offset_data_list)
for line in angle_offset_data_list:
    csv_angle_offset_writer.writerow(line)
csv_angle_offset_data_list.close()

csv_10_cone_data_list = open(path + "2020_12_11_datablock_10degcone.csv", 'w',
                             newline="")  # create a csv file to collect spider sight angles, speed, distance, etc
csv_10_cone_writer = csv.writer(csv_10_cone_data_list)
for line in cone_10_data_list:
    csv_10_cone_writer.writerow(line)
csv_10_cone_data_list.close()

csv_28_cone_data_list = open(path + "2020_12_11_datablock_28degcone.csv", 'w',
                             newline="")  # create a csv file to collect spider sight angles, speed, distance, etc
csv_28_cone_writer = csv.writer(csv_28_cone_data_list)
for line in cone_28_data_list:
    csv_28_cone_writer.writerow(line)
csv_28_cone_data_list.close()

csv_56_cone_data_list = open(path + "2020_12_11_datablock_56degcone.csv", 'w',
                             newline="")  # create a csv file to collect spider sight angles, speed, distance, etc
csv_56_cone_writer = csv.writer(csv_56_cone_data_list)
for line in cone_56_data_list:
    csv_56_cone_writer.writerow(line)
csv_56_cone_data_list.close()
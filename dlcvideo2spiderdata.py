import numpy as np
import cv2 as cv
import csv
import math

'''
Last changed: 2021-02-22
README

Only run this code if you've already run prep_dlcvideo2spiderdata_files.py!

Note: Spider position is referred to as "midpoint"

Use this code to:
1. Generate a vector csv for each video with a list of position and sightline unit vectors for each frame
2. Generate a sightlines csv for each video with distance, sightline angle offset, view cone, etc. for each frame

How:
1. Make sure this code is in the same folder as ALL the MOVs, csvs, and post processing code
2. Replace the path variable with the path to the folder with everything
3. Run dlcvideo2spiderdata.py 
4. Run sightline_processing.py in the same folder. 
'''

path = "C:/Users/ShambleLab/Documents/condaDLCenv/2020_04_20_salticus/videos/" #<-- THIS IS THE ONLY VARIABLE YOU NEED TO CHANGE, FOLDERPATH TO THE 1) CSV FILE WITH THE LIST OF EVERYTHING, 2) THE DLC OUTPUTS, 3) VIDEOS
file_list_read = open(path+"dlc_video_names.csv") #add name of the specific csv file with outputs/video here
file_list = csv.reader(file_list_read, delimiter = ',') #read the csv file

for file_row in file_list:
    #iterate through the rows of the csv file and pull the dlc output name, video name, and scale factor
    num_columns = len(file_row)
    dlc_file_name = file_row[0]
    video_file_name = file_row[1]
    scale_factor = float(file_row[2])

    #print the files you are currently analyzing and processing
    print(dlc_file_name)
    print(video_file_name)

    '''
    VECTOR PARSING CODE
    This code takes the DLC output csv, extracts vector data, and produces a vector csv with:
    1) Head midpoint x
    2) Head midpoint y
    3) Spider orientation unit vector x
    4) Spider orientation unit vector y
    
    The vector csv will be your DLC file name + _vectors
    '''

    #load the dlc output csv and turn all the tracked points into floats
    dlc_output = np.loadtxt(path+dlc_file_name, delimiter = ",", skiprows = 3)
    dlc_output = dlc_output.astype(float)
    num_lines = len(dlc_output)

    #CALCULATE AVERAGE MIDPOINT POSITION AND F-LH-RH and P-LH-RH ANGLES FOR BACK-CALC PURPOSES

    #create an array to hold headpoints that we are confident in
    #this will be the points we use to calculate average positions/angles
    A = np.array([])
    first = 1
    iterate = -1

    #create a csv file to hold all the calculated vectors
    vector_output = open(path+dlc_file_name[:-4]+"_vectors.csv", 'w', newline = "")
    vector_writer = csv.writer(vector_output)

    #iterate through the dlc tracking points and add the frontspot, lheadspot, rheadspot, and pedicel point sets that have high confidence
    for line in dlc_output:
        if ((line[3]>0.98) and (line[6]>0.98) and (line[9]>0.98) and (line[12]>0.98)):
            if first:
                A = np.append(A, [line[1], line[2], line[4], line[5], line[7], line[8], line[10], line[11]], axis=0)
                first = 0
            else:
                A = np.vstack((A, [line[1], line[2], line[4], line[5], line[7], line[8], line[10], line[11]]))

    #calculate the distance between the lh-rh, p-f, and lh-p
    lhrh_distance_list = np.sqrt(((A[:,4]-A[:,2])**2)+(((A[:,5]-A[:,3])**2)))
    pf_distance_list = np.sqrt(((A[:,6]-A[:,0])**2)+(((A[:,7]-A[:,1])**2)))
    lhp_distance_list = np.sqrt(((A[:,6]-A[:,2])**2)+(((A[:,7]-A[:,3])**2)))

    #calculate the average distance between the lh-rh, p-f, and lh-p
    lhrh_distance_average = np.mean(lhrh_distance_list)
    pf_distance_average = np.mean(pf_distance_list)
    lhp_distance_average = np.mean(lhp_distance_list)

    #calculate the lh-f, lh-rh, and lh-p vectors and transpose them into columns
    lhf_vector_list_linear = np.array([A[:,0]-A[:,2], A[:,1]-A[:,3]])
    lhf_vector_list = lhf_vector_list_linear.transpose()
    lhrh_vector_list_linear = np.array([A[:,4]-A[:,2], A[:,5]-A[:,3]])
    lhrh_vector_list = lhrh_vector_list_linear.transpose()
    lhp_vector_list_linear = np.array([A[:,6]-A[:,2], A[:,7]-A[:,3]])
    lhp_vector_list = lhp_vector_list_linear.transpose()

    #find the orthogonal projection of the rheadspot onto the p-f line
    '''
    m1_list = np.array([(A[:,1]-A[:,7])/(A[:,0]-A[:,6])])
    m2_list = -1.0/m1_list
    headspot_midpoint_x_list = (m2_list*A[:,4]-A[:,5]+A[:,7]-m1_list*A[:,6])/(m2_list-m1_list)
    headspot_midpoint_y_list = m2_list*(headspot_midpoint_x_list-A[:,4])+A[:,5]
    headspot_midpoint_list_linear = np.array([headspot_midpoint_x_list, headspot_midpoint_y_list])
    headspot_midpoint_list = headspot_midpoint_list_linear.transpose()
    '''
    fp_vector_list_linear = np.array([A[:,6]-A[:,0], A[:,7]-A[:,1]])
    fp_vector_list = fp_vector_list_linear.transpose()
    frh_vector_list_linear = np.array([A[:,4]-A[:,0], A[:,5]-A[:,1]])
    frh_vector_list = frh_vector_list_linear.transpose()
    projected_vector_x_list = np.zeros(len(fp_vector_list))
    projected_vector_y_list = np.zeros(len(fp_vector_list))
    for n in range(len(fp_vector_list)):
        projected_vector_x_list[n], projected_vector_y_list[n] = (np.dot(fp_vector_list[n],frh_vector_list[n])/np.dot(fp_vector_list[n],fp_vector_list[n]))*fp_vector_list[n]
    projected_vector_list_linear = np.array([projected_vector_x_list, projected_vector_y_list])
    projected_vector_list = projected_vector_list_linear.transpose()
    headspot_midpoint_list_linear = np.array([[A[:,0]+projected_vector_list[:,0]], [A[:,1]+projected_vector_list[:,1]]])
    headspot_midpoint_list = headspot_midpoint_list_linear.transpose()

    #use that orthogonal projection to calculate the average midpoint position on the p-f line
    #output this as a ratio of the p-f length from the frontspot
    midpoint_ratio_list = (np.sqrt(((headspot_midpoint_list[:,0][:,0]-A[:,0])**2)+((headspot_midpoint_list[:,0][:,1]-A[:,1])**2)))/pf_distance_list
    midpoint_ratio_list_average = np.mean(midpoint_ratio_list)

    #create arrays to collect the angles between f-lh-rh and p-lh-rh
    flhrh_angle_list = np.zeros(len(lhrh_distance_list))
    plhrh_angle_list = np.zeros(len(pf_distance_list))

    #collect the f-lh-rh and p-lh-rh angles in their respective arrays
    for i in range(len(lhrh_distance_list)):
        flhrh_angle_list[i] = np.arccos((np.dot(lhf_vector_list[i,:], lhrh_vector_list[i,:])/(np.linalg.norm(lhf_vector_list[i,:])*np.linalg.norm(lhrh_vector_list[i,:]))))

    for i in range(len(lhrh_distance_list)):
        plhrh_angle_list[i] = np.arccos(np.dot(lhp_vector_list[i,:], lhrh_vector_list[i,:])/(np.linalg.norm(lhp_vector_list[i,:])*np.linalg.norm(lhrh_vector_list[i,:])))

    #calculate the mean f-lh-rh and mean p-lh-rh angles
    flhrh_angle = np.mean(flhrh_angle_list)
    plhrh_angle = np.mean(plhrh_angle_list)

    #CALCULATE SPIDER POINT VECTORS FROM DLC TRACKED POINTS

    #create an array to collect vectors [base_x, base_y, tip_x, tip_y]
    spider_point_list = np.zeros((num_lines, 4))

    #iterate through the dlc tracking points
    for line in dlc_output:
        iterate+= 1 #keep track of the video frame

        #pull frontspot, lheadspot, rheadspot, and pedicel data points
        frontspotx = line[1]
        frontspoty = line[2]
        frontspotp = line[3]
        lheadspotx = line[4]
        lheadspoty = line[5]
        lheadspotp = line[6]
        rheadspotx = line[7]
        rheadspoty = line[8]
        rheadspotp = line[9]
        pedicelx = line[10]
        pedicely = line[11]
        pedicelp = line[12]

        #if f, lh, rh, and p are all confident, just take the pf vector and calculate the midpoint from lh and rh
        if ((frontspotp>0.98) and (lheadspotp>0.98) and (rheadspotp>0.98) and (pedicelp>0.98)):
            headspot_midpoint = [(lheadspotx+rheadspotx)/2,(lheadspoty+rheadspoty)/2]

            pf_vector = [frontspotx-pedicelx, frontspoty-pedicely]
            pf_vector_unit = pf_vector/np.linalg.norm(pf_vector)

            spider_point_list[iterate, 0] = headspot_midpoint[0]
            spider_point_list[iterate, 1] = headspot_midpoint[1]
            spider_point_list[iterate, 2] = pf_vector_unit[0]
            spider_point_list[iterate, 3] = pf_vector_unit[1]

        #if only rh, lh, and f are confident, then calculate the midpoint from lh and rh and calculate the vector orthogonal to lh-rh for the pf vector
        elif ((rheadspotp>0.98) & (lheadspotp>0.98) & (frontspotp>0.98) & (pedicelp<0.98)):
            headspot_midpoint = [(lheadspotx + rheadspotx) / 2, (lheadspoty + rheadspoty) / 2]

            lhrh_vector = [rheadspotx - lheadspotx, rheadspoty - lheadspoty]
            hmf_vector = [frontspotx - headspot_midpoint[0], frontspoty - headspot_midpoint[1]]

            lhrh_vector_unit = lhrh_vector / np.linalg.norm(lhrh_vector)
            hmf_vector_unit = hmf_vector / np.linalg.norm(hmf_vector)

            #find the orthogonal vector to the lh-rh vector
            lhrh_vector_unit_orth = [
                lhrh_vector_unit[0] * math.cos(math.radians(90)) - lhrh_vector_unit[1] * math.sin(math.radians(90)),
                lhrh_vector_unit[0] * math.sin(math.radians(90)) + lhrh_vector_unit[1] * math.cos(math.radians(90))]
            average_vector_unit_orth = [lhrh_vector_unit_orth[0], lhrh_vector_unit_orth[1]]

            spider_point_list[iterate, 0] = headspot_midpoint[0]
            spider_point_list[iterate, 1] = headspot_midpoint[1]
            spider_point_list[iterate, 2] = average_vector_unit_orth[0]
            spider_point_list[iterate, 3] = average_vector_unit_orth[1]

        #if f, p, and one of lh and rh are confident, find the midpoint as the orthogonal projection of either rh and lh on p-f and take the p-f vector
        elif ((frontspotp>0.98) & (pedicelp>0.98)) & (((rheadspotp>0.98) & (lheadspotp<0.98) & (lheadspotp>0.9))|((lheadspotp>0.98) & (rheadspotp<0.98) & (rheadspotp>0.9))):
            if (rheadspotp>0.9):
                m1 = (frontspoty-pedicely)/(frontspotx-pedicelx)
                m2 = -1/m1
                headspot_midpoint_x = (m2 * rheadspotx - rheadspoty + pedicely - m1 * pedicelx) / (m2 - m1)
                headspot_midpoint_y = m2 * (headspot_midpoint_x - rheadspotx) + rheadspoty
                headspot_midpoint = [headspot_midpoint_x, headspot_midpoint_y]

            if (lheadspotp>0.9):
                m1 = (frontspoty - pedicely) / (frontspotx - pedicelx)
                m2 = -1 / m1
                headspot_midpoint_x = (m2 * lheadspotx - lheadspoty + pedicely - m1 * pedicelx) / (m2 - m1)
                headspot_midpoint_y = m2 * (headspot_midpoint_x - lheadspotx) + lheadspoty
                headspot_midpoint = [headspot_midpoint_x, headspot_midpoint_y]

            pf_vector = [frontspotx - pedicelx, frontspoty - pedicely]
            pf_vector_unit = pf_vector / np.linalg.norm(pf_vector)

            spider_point_list[iterate, 0] = headspot_midpoint[0]
            spider_point_list[iterate, 1] = headspot_midpoint[1]
            spider_point_list[iterate, 2] = pf_vector_unit[0]
            spider_point_list[iterate, 3] = pf_vector_unit[1]

        #if f and p are confident, take the p-f vector and calculate the midpoint using the midpoint average ratio found earlier
        elif ((frontspotp>0.98) & (pedicelp>0.98) & (rheadspotp<0.98) & (lheadspotp<0.98)):
            pf_vector = [frontspotx - pedicelx, frontspoty - pedicely]
            headspot_midpoint = [pedicelx + midpoint_ratio_list_average*pf_vector[0], pedicely + midpoint_ratio_list_average*pf_vector[1]]
            pf_vector_unit = pf_vector/np.linalg.norm(pf_vector)

            spider_point_list[iterate, 0] = headspot_midpoint[0]
            spider_point_list[iterate, 1] = headspot_midpoint[1]
            spider_point_list[iterate, 2] = pf_vector_unit[0]
            spider_point_list[iterate, 3] = pf_vector_unit[1]

        #if f and one of lh or rh are confident, reconstruct p from average angles found earlier and use the p-f vector and find the midpoint using the midpoint average ratio found earlier
        elif ((frontspotp>0.98) & (pedicelp<0.98) & (((rheadspotp>0.98) & (lheadspotp<0.98) & (lheadspotp>0.9))|((lheadspotp>0.98) & (rheadspotp<0.98) & (rheadspotp>0.9)))):
            if (rheadspotp>0.98):
                rf_vector = [frontspotx - rheadspotx, frontspoty - rheadspoty]
                rf_vector_unit = rf_vector/np.linalg.norm(rf_vector)
                rf_vector_rotated = [rf_vector_unit[0]*math.cos(flhrh_angle + plhrh_angle) - rf_vector_unit[1]*math.sin(flhrh_angle + plhrh_angle), rf_vector_unit[0]*math.sin(flhrh_angle + plhrh_angle) + rf_vector_unit[1]*math.cos(flhrh_angle + plhrh_angle)]
                p_reconstructed = [rheadspotx + lhp_distance_average*rf_vector_rotated[0], rheadspoty + lhp_distance_average*rf_vector_rotated[1]]
                pf_vector = [frontspotx - p_reconstructed[0], frontspoty - p_reconstructed[1]]
                pf_distance = math.sqrt(((frontspotx - p_reconstructed[0])**2) + ((frontspoty - p_reconstructed[1])**2))
                headspot_midpoint = [p_reconstructed[0] + midpoint_ratio_list_average*pf_vector[0], p_reconstructed[1] + midpoint_ratio_list_average*pf_vector[1]]
                pf_vector_unit = pf_vector/np.linalg.norm(pf_vector)

            elif (lheadspotp>0.98):
                lf_vector = [frontspotx - lheadspotx, frontspoty - lheadspoty]
                lf_vector_unit = lf_vector/np.linalg.norm(lf_vector)
                lf_vector_rotated = [lf_vector_unit[0]*math.cos(flhrh_angle + plhrh_angle) - lf_vector_unit[1]*math.sin(flhrh_angle + plhrh_angle), lf_vector_unit[0]*math.sin(flhrh_angle + plhrh_angle) + lf_vector_unit[1]*math.cos(flhrh_angle + plhrh_angle)]
                p_reconstructed = [lheadspotx + lhp_distance_average*lf_vector_rotated[0], lheadspoty + lhp_distance_average*lf_vector_rotated[1]]
                pf_vector = [frontspotx - p_reconstructed[0], frontspoty - p_reconstructed[1]]
                pf_distance = math.sqrt(((frontspotx - p_reconstructed[0])**2) + ((frontspoty - p_reconstructed[1])**2))
                headspot_midpoint = [p_reconstructed[0] + midpoint_ratio_list_average*pf_vector[0], p_reconstructed[1] + midpoint_ratio_list_average*pf_vector[1]]
                pf_vector_unit = pf_vector/np.linalg.norm(pf_vector)

            spider_point_list[iterate, 0] = headspot_midpoint[0]
            spider_point_list[iterate, 1] = headspot_midpoint[1]
            spider_point_list[iterate, 2] = pf_vector_unit[0]
            spider_point_list[iterate, 3] = pf_vector_unit[1]

        #write the spider_point_list row to the csv file (this will be all zeros if none of the points are confident enough to fulfill the former conditions)
        record_row = [str(spider_point_list[iterate, 0]), str(spider_point_list[iterate, 1]), str(spider_point_list[iterate, 2]), str(spider_point_list[iterate, 3])]
        vector_writer.writerow(record_row)

    vector_output.close() #close the vector csv file

    '''
    SPIDER SIGHTLINE/POSITION CODE
    This code uses the vector csv you just produced in combination with the spider video to give you a csv with:
    1) Frame #
    2) Spider midpoint x
    3) Spider midpoint y
    4) Angle offset between the spider sightline and the line from the spider midpoint to the predator center
    5) Distance from the spider midpoint to the predator center (px)
    6) Distance from the spider midpoint to the predator center (mm)
    7) Spider average midpoint speed (calculated as of the last marked frame) (pixel/frame)
    8) Spider average midpoint speed (calculated as of the last marked frame) (mm/s)
    9) mm/px conversion factor
    10) predator center x
    11) predator center y
    12) predator bounding box ULH x
    13) predator bounding box ULH y
    14) predator bounding box URH x
    15) predator bounding box URH y
    16) predator bounding box LLH x
    17) predator bounding box LLH y
    18) predator bounding box LRH x
    19) predator bounding box LRH y
    20) 10deg view cone predator detect 
    21) 28deg view cone predator detect
    22) 56deg view cone predator detect
    23) 0-2pi rad spider view angle
    
    The sightline csv name will be your DLC file name + _sightline
    '''

    is_looking = False
    cap = cv.VideoCapture(path + video_file_name) #open the video file
    dlc_vectors = np.loadtxt(path + dlc_file_name[:-4] + "_vectors.csv", delimiter=",") #open the vector csv file
    dlc_vectors = dlc_vectors.astype(float) #read the vector csv file in floats
    frame_number = -1 #frame counter
    loop_number = 0
    framerate = cap.get(cv.CAP_PROP_FPS) #grab framerate
    frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH) #grab frame width
    csv_output = open(path+ dlc_file_name[:-4] + "_sightline.csv", 'w', newline="") #create a csv file to collect spider sight angles, speed, distance, etc
    csv_writer = csv.writer(csv_output)

    # #this code is only for spider videos that start with a title card
    # #pressing "Q" allows the users to start predator seeking/labeling after the title card is gone
    # while True:
    #     ret, frame = cap.read()
    #     frame_number += 1
    #     if ret == False:
    #         break
    #     resize_frame = cv.resize(frame, (960, 530))
    #     cv.imshow("frame", resize_frame)
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         break

    #ADD PREDATOR LENGTH VARIABLE
    mm_per_pixel = scale_factor

    while True:
        if frame_number <0: #with the first frame, grab the predator location data and the frame data
            ret, frame = cap.read()  # read the first video frame
            frame_number += 1
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # grey out the frame
            fr_height, fr_width = gray_frame.shape
            var, new_frame = cv.threshold(gray_frame, 85, 255, cv.THRESH_BINARY_INV) #threshold and inverse the frame so that the predator is distinct from its surroundings
            cv.rectangle(new_frame, (int(fr_width / 3), fr_height), (fr_width, 0), 0,
                         -1)  # black out the right 2/3 of the frame to avoid capturing anything on the launch platform
            cv.rectangle(new_frame, (0, fr_height), (fr_width, fr_height - 250), 0,
                         -1)  # black out the bottom of the frame to avoid capturing random shadows
            cv.rectangle(new_frame, (0, 0), (fr_width, 250), 0,
                         -1)  # black out the top of the frame to avoid capturing random shadows
            contours, _ = cv.findContours(new_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[
                          -2:]  # find contours of the predator, returns last 2 vals (makes compatable across CV versions)
            contour = np.vstack(contours)  # combine the contours
            hull = cv.convexHull(contour)  # draw a polygon around the predator
            M = cv.moments(hull)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            hull_center = (cX, cY)  # grab the hull center
            rx, ry, rw, rh = cv.boundingRect(contour)  # find a bounding rectangle around the predator

        else: #with following frames, just grab the frame and frame number
            ret, frame = cap.read() #read the video frame by frame
            frame_number += 1

        color = (255, 0, 0)
        angle_offset = None
        speed = None
        if ret == False:
            break

        #if the dlc vector for the corresponding frame exists:
        #1) calculate the distance from the spider's midpoint to the predator's center
        #2) extend out the spider sightline vector
        #3) calculate the offset angle from the spider's current sightline to the line from the spider midpoint to the predator
        #4) calculate whether a 10/28/56 view cone intersects with the predator hull
        #5) calculate the spider sightline angle in radians
        #6) calculate the speed of the spider's midpoint
        if (dlc_vectors[frame_number][2] or dlc_vectors[frame_number][3]) != 0:
            loop_number += 1

            # calculate the extended end of the vector
            base = (dlc_vectors[frame_number][0], dlc_vectors[frame_number][1]) #pull the vector base/midpoint
            base_np = np.array(base) #grab the vector base/midpoint as a numpy array
            base_to_arrow_np = np.array([dlc_vectors[frame_number][2], dlc_vectors[frame_number][3]]) #pull the vector as a numpy array
            extended = (int(base[0] + dlc_vectors[frame_number][2] * 3000), int(base[1] + dlc_vectors[frame_number][3] * 3000)) #extend out the vector
            hull_np = np.array(hull_center) #grab the hull center as a numpy array
            base_to_hull_np = hull_np - base_np
            base_to_hull_distance = np.linalg.norm(base_to_hull_np) #calculate distance from spider midpoint to predator hull center
            angle_offset = np.arccos(np.dot(base_to_hull_np, base_to_arrow_np) / (
                        np.linalg.norm(base_to_hull_np) * np.linalg.norm(base_to_arrow_np))) #calculate the angle between the extended spider vector-spider midpoint-predator center
            base_int = (int(dlc_vectors[frame_number][0]), int(dlc_vectors[frame_number][1])) #opencv can only plot ints
            cv.line(frame, base_int, extended, (255, 0, 0), 5) #draw the extended vector

            #draw 10/28/56deg view cones and find if they intersect with the predator hull
            view_angles = [0.0873, 0.2443, 0.4887] #5/14/28deg (half of view cone because we are rotating the spider sightline vector cw and ccw)
            view_cone_intersect = [] #collect whether the current view cones intersect with the predator hull: 0: 10, 1: 28, 2: 56
            for angle in view_angles:
                cc_rotation = np.array([base_to_arrow_np[0] * math.cos(angle) - base_to_arrow_np[1] * math.sin(angle),
                                        base_to_arrow_np[0] * math.sin(angle) + base_to_arrow_np[1] * math.cos(angle)]) #rotate sightline vector ccw
                cw_rotation = np.array([base_to_arrow_np[0] * math.cos(angle) + base_to_arrow_np[1] * math.sin(angle),
                                        -base_to_arrow_np[0] * math.sin(angle) + base_to_arrow_np[1] * math.cos(angle)]) #rotate sightline vector cw
                view_cone = np.array(
                    [[base[0], base[1]], [base[0] + cc_rotation[0] * 3000, base[1] + cc_rotation[1] * 3000],
                     [base[0] + cw_rotation[0] * 3000, base[1] + cw_rotation[1] * 3000]], dtype=np.int32) #all the points of the view cone (base, cw extended point, ccw extended point)
                blank = np.zeros(frame.shape[0:2]) #make a image array of all zeros the size of the frame
                img1 = cv.drawContours(blank.copy(), [hull], -1, 1, cv.FILLED) #make an image array of the frame with 1's in the hull space
                img2 = cv.drawContours(blank.copy(), [view_cone], -1, 1, cv.FILLED) #make an image array of the frame with 1's in the view cone space
                verdict = int(np.sum(img1 * img2) > 0) #find if there's any intersection between the hull space and the view cone
                view_cone_intersect.append(verdict) #add the verdict to the view_cone_intersect list

            # calculate spider sightline angle in radians with (1, 0) being 0rad, angle increases ccw
            arctan2 = math.atan2(base_to_arrow_np[1], base_to_arrow_np[0])
            if arctan2<0:
                overall_angle = abs(arctan2)
            else:
                overall_angle = 2*math.pi-abs(arctan2)

            #if this set of vectors is not the first set, calculate the spider speed using the current midpoint and the last recorded midpoint
            if (loop_number > 1):
                movement_distance = np.linalg.norm(base_np - [previous_write_array_np[1], previous_write_array_np[2]]) #calculate the spider movement distance
                movement_distance_mm = movement_distance * mm_per_pixel #convert to mm
                movement_speed_px_per_f = movement_distance / (frame_number - previous_write_array_np[0]) #calculate speed in pixels per frame
                movement_speed_mm_per_s = movement_distance_mm / ((frame_number - previous_write_array_np[0]) / framerate) #calculate speed in mm per s
                speed = movement_speed_mm_per_s
                write_array = [str(frame_number + 1), str(dlc_vectors[frame_number][0]), str(dlc_vectors[frame_number][1]),
                               str(angle_offset), str(base_to_hull_distance), str(base_to_hull_distance * mm_per_pixel),
                               str(movement_speed_px_per_f), str(movement_speed_mm_per_s), str(mm_per_pixel), str(cX), str(cY),
                               str(rx), str(ry), str(rx+rw), str(ry), str(rx), str(ry+rh), str(rx+rw), str(ry+rh), str(view_cone_intersect[0]),
                               str(view_cone_intersect[1]), str(view_cone_intersect[2]), str(overall_angle)] #record the frame number (plus one because of zero indexing), midpoint coordinates, angle offset, spider distance (mm and px), spider speed (mm/s and px/f), predator box coordinates, and view cone intersect verdicts
                csv_writer.writerow(write_array) #write to the sightline csv

            #if this is the first set of vectors, set the spider speed to zero
            else:
                write_array = [str(frame_number + 1), str(dlc_vectors[frame_number][0]), str(dlc_vectors[frame_number][1]),
                               str(angle_offset), str(base_to_hull_distance), str(base_to_hull_distance * mm_per_pixel),
                               str(0), str(0), str(mm_per_pixel), str(cX), str(cY), str(rx), str(ry), str(rx+rw), str(ry), str(rx), str(ry+rh), str(rx+rw), str(ry+rh),
                               str(view_cone_intersect[0]), str(view_cone_intersect[1]), str(view_cone_intersect[2]), str(overall_angle)] #record the frame number (plus one because of zero indexing), midpoint coordinates, angle offset, spider distance (mm and px), predator box coordinates, and view cone intersect verdicts
                csv_writer.writerow(write_array) #write to the sightline csv

            previous_write_array_np = np.array([frame_number, dlc_vectors[frame_number][0], dlc_vectors[frame_number][1]]) #save a snapshot of the current frame's set of data to use to calculate spider speed with the next frame of vector data

            #change the color of the overlay video text if the spider sightline angle goes within 20deg of the predator's center
            if angle_offset < 0.348:
                is_looking = True
                color = (0, 255, 0)
            else:
                is_looking = False
                color = (0, 0, 255)
        '''
        #VISUALIZATION
        #comment this section out if you want to not have the visualization
        cv.drawContours(frame, [hull], -1, (0, 255, 0), 5) #draw the polygon around the predator
        cv.circle(frame, hull_center, 3, (0, 0, 255), -1) #mark the predator center
        cv.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2) #draw a rectangle around the predator
        cv.putText(frame, str(angle_offset), (0, 100), cv.FONT_HERSHEY_TRIPLEX, 2, color) #write the angle offset
        cv.putText(frame, str(speed), (0, 200), cv.FONT_HERSHEY_TRIPLEX, 2, color) #write the spider speed (mm/s) below the angle offset
        resize_frame = cv.resize(frame, (960, 530)) #resize the video frame so it doesn't take up the whole computer screen
        cv.imshow("frame", resize_frame) #show the final frame!
        if cv.waitKey(1) & 0xFF == ord('q'): #video ends if the user hits "q"
            break
        '''

    cv.destroyAllWindows() #close all the video windows after processing each video
    cap.release() #close the video file
    csv_output.close() #close the sightline csv file
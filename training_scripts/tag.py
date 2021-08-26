import numpy as np
import cv2
import os, glob, time
from random import sample

label_colors = {1:(0,0,255),
          2:(255,0,0),
          3:(0,255,0),
          4:(255,0,255),
          5:(0,255,255),
          6:(255,255,0)}

def tag_objects(frameId, image, videoFile, label_list):
    '''
    Initializes and completes the tagging process for a single frame.
    Inputs:
        frameId - (int) frame number
        image - numpy array
        videoFile - name of the video from which the frame was taken
    Returns:
        output_dict - dictionary of features which can then be serialized to a
                      tf_example object
    '''
    videoFile_pathless = videoFile.split("/")[-1]
    output_dict = {'video': str(videoFile_pathless), 'source_id': str(frameId),
                   'image_width':None, 'image_height':None,
                    'xmins':[], 'xmaxs':[], 'ymins':[], 'ymaxs':[], 'classes':[],
                    'classes_text':[]}
    output_dict['image_height']=image.shape[0]
    output_dict['image_width']=image.shape[1]
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.startWindowThread()
    cv2.namedWindow("image")

    while True:
        done = False
        # display the image and allow bounding box selection
        cv2.imshow("image", image)
        #cv2.moveWindow("image", -305, -1000)
        ROI = cv2.selectROI("image", image, showCrosshair=False)
        cv2.destroyWindow("ROI selector")
        cv2.waitKey(100)
        corner1 = (ROI[0], ROI[1])
        corner2 = (ROI[0]+ROI[2], ROI[1]+ROI[3])
        cv2.rectangle(image, corner1, corner2, (255, 255, 255), 1)
        cv2.imshow("image", image)
        key = cv2.waitKey(0) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

        # If the 's' key is pressed, break from the loop and return an empty dictionary.
        elif key == ord("s"):
            output_dict = {}
            break
        # if the 'q' key is pressed, break from the loop
        elif key == ord("q"):
            break

        for i, (label, letter) in enumerate(label_list):
            if key == ord(letter.lower()) or key == ord(letter.upper()):
                if i >= len(label_colors):
                    color = (255,255,0)
                else:
                    color = label_colors[i+1]
                image = clone.copy()
                cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), color, 1)
                update_dict_coords(output_dict, i+1, label, corner1, corner2)
                cv2.imshow('image', image)
                if key == ord(letter.upper()):
                    done=True
                    break
                clone = image.copy()
        if done: break



    # close all open windows
    cv2.destroyAllWindows()
    cv2.waitKey(10)
    return output_dict


def createTrackerByName(trackerType):
    '''
    Takes a tracker type and creates a corresponding tracker object.
    '''
    trackerTypes = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker



def update_tracked(output_dict, boxes, clone, next_frameId):
    '''
    Once a frame has been tagged and the tagged objects have been tracked in a
    subsequent frame, this function produces a new output_dict object for the
    new frame.
    Inputs:
        output_dict - dictionary of features for the frame that was tagged
        boxes - list of boxes provided by the multiTracker for new coordinates
                of the original tagged objects.
        clone - numpy array of the image for the new frame
        next_frameId - number of the frame for the tracked image
    Returns:
        new_output_dict - dictionary of features for the tracked frame
    '''
    new_output_dict = {}
    new_output_dict['video'] = output_dict['video']
    new_output_dict['image_width'] = output_dict['image_width']
    new_output_dict['image_height'] = output_dict['image_height']
    new_output_dict['classes'] = output_dict['classes']
    new_output_dict['classes_text'] = output_dict['classes_text']

    updated_xmins = []
    updated_xmaxs = []
    updated_ymins = []
    updated_ymaxs = []
    for box in boxes:
        updated_xmins.append(box[0])
        updated_ymins.append(box[1])
        updated_xmaxs.append(box[0] + box[2])
        updated_ymaxs.append(box[1] + box[3])
    new_output_dict['xmins'] = updated_xmins
    new_output_dict['ymins'] = updated_ymins
    new_output_dict['xmaxs'] = updated_xmaxs
    new_output_dict['ymaxs'] = updated_ymaxs
    new_output_dict['source_id'] = str(next_frameId)

    return new_output_dict


def update_dict_coords(output_dict, label, class_text, corner1, corner2):
    '''
    Inputs:
        output_dict - dictionary of features to update
        label - (int) label of the new object to update
        class_text - (string) label text of the new object to update ('car')
        corner1 and corner2 - (tuples) x and y coordinates of opposite corners
            of the bounding box for the object to update - can be any two
            opposite corners.
    Returns nothing - modifies output_dict in place.
    '''
    bb = order_bounding_box(corner1, corner2)
    output_dict['xmins'].append(bb[0][0])
    output_dict['ymins'].append(bb[0][1])
    output_dict['xmaxs'].append(bb[1][0])
    output_dict['ymaxs'].append(bb[1][1])
    output_dict['classes'].append(label)
    output_dict['classes_text'].append(class_text)


def order_bounding_box(corner1, corner2):
    '''
    Takes as input any two corners of a rectangle and returns two pairs of coordinates -
    upper left and lower right.
    '''
    upper_left = (min(corner1[0], corner2[0]), min(corner1[1], corner2[1]))
    lower_right = (max(corner1[0], corner2[0]), max(corner1[1], corner2[1]))
    return [upper_left, lower_right]

def print_tagging_instructions(label_list):
    '''
    Prints instructions for tagging objects.
    '''

    print("*******************************************************************")
    print("Instructions for tagging frames")
    print("1. Drag a bounding box around the visible part of the object.")
    print('2. Press "enter" to accept bounding box.  Press "c" to redo.')
    print('Bounding box should turn white when accepted.')
    print("3.  Once the bounding box is white, press")
    for (label, letter) in label_list:
        print("'{}' for {}".format(letter, label))
    print("'r' to redo")
    print("'q' to quit the frame, and")
    print("'s' to skip the frame without saving a record.")
    print("4.  If you tagged a person, car, truck, or bus, the bounding box should change color.")
    print("5.  To quit before the number of frames specified has been tagged, press Ctrl-C.")
    print("6.  Tag every object (person, car, truck or bus) in the frame")
    print("7.  Make sure bounding box covers only the visible parts of the object.")
    print("8.  Don't tag objects that are mostly hidden.")
    print("*******************************************************************")

def print_summary(count, output_path):
    '''
    Prints a summary once tagging is finished.
    '''
    print("*******************************************************************")
    print("Summary:")
    print("You tagged {} frames.".format(count))
    print("*******************************************************************")

def sample_bins(frames_tagged, num_records, num_bins, frame_lim_low, frame_lim_high):
    #Create bins
    bins = []
    frame_set = set()
    width = (frame_lim_high + 1 - frame_lim_low)/num_bins
    divs = np.linspace(frame_lim_low, frame_lim_high + 1, num_bins + 1)
    for i in range(num_bins):
        bins.append([x for x in range(int(divs[i]), int(divs[i+1]))])
    #Count number of frames in each bin.
    end_frames_per_bin = int((len(frames_tagged) + 5*num_records)/num_bins)
    for bin in bins:
        n_bin = int((end_frames_per_bin - len(set(bin).intersection(frames_tagged)))/5)
        n_bin = n_bin if n_bin>0 else 1
        frames_bin = set(sample(range(bin[0],bin[-1]),n_bin))
        frame_set = frame_set.union(frames_bin)
    return frame_set
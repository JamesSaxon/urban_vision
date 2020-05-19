import numpy as np
import cv2
import os, glob, time


def tag_objects(frameId, image, videoFile):
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
    output_dict = {'video': str(videoFile), 'source_id': str(frameId), 'image_width':None, 'image_height':None,
                    'xmins':[], 'xmaxs':[], 'ymins':[], 'ymaxs':[], 'classes':[],
                    'classes_text':[]}
    output_dict['image_height']=image.shape[0]
    output_dict['image_width']=image.shape[1]
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.startWindowThread()
    cv2.namedWindow("image")

    while True:
        # display the image and allow bounding box selection
        cv2.imshow("image", image)
        cv2.moveWindow("image", -305, -1000)
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

        # if the 'p' key is pressed, save coordinates, 'person', and draw rectangle in green.
        if key == ord("p"):
            image = clone.copy()
            cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 255, 0), 1)
            update_dict_coords(output_dict, 1, 'person', corner1, corner2)
            cv2.imshow('image', image)
            clone = image.copy()

        # if the 'b' key is pressed, save coordinates, 'bus', and draw rectangle in red.
        if key == ord("b"):
            image = clone.copy()
            cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 0, 255), 1)
            update_dict_coords(output_dict, 3, 'bus', corner1, corner2)
            cv2.imshow('image', image)
            clone = image.copy()

        # if the 'c' key is pressed, save coordinates, 'car', and draw rectangle in yellow.
        if key == ord("c"):
            image = clone.copy()
            cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 255, 255), 1)
            update_dict_coords(output_dict, 2, 'car', corner1, corner2)
            cv2.imshow('image', image)
            clone = image.copy()

        # if the 'q' key is pressed, break from the loop
        elif key == ord("q"):
            break

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

def print_tagging_instructions():
    '''
    Prints instructions for tagging objects.
    '''
    print("*******************************************************************")
    print("Instructions for tagging frames")
    print("1. Drag a bounding box around the visible part of the object.")
    print('2. Press "enter" to accept bounding box.  Press "c" to redo.')
    print('Bounding box should turn white when accepted.')
    print("3.  Once the bounding box is white, press 'p' for person, 'c' for car,")
    print("'b' for bus, 'r' to redo, and 'q' to quit the frame.")
    print("4.  If you tagged a person, car, or bus, the bounding box should change color.")
    print("5.  To quit before the number of frames specified has been tagged, press Ctrl-C.")
    print("6.  Tag every object (person, car, or bus) in the frame")
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

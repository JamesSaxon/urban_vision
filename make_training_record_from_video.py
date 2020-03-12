import cv2
import math
import json
import tensorflow as tf
import numpy as np
from object_detection.utils import dataset_util
import os, glob
from PIL import Image
import pathlib
from random import sample
import argparse


num_records = 100
save_tagged_image = False
write_to_json = False
videoFile = "keller_a_03122020_2.mov"
video_duration = 2*60

output_path = "/Users/amandawhaley/Projects/UrbanVision/train_" + \
                videoFile.split(".")[0] + ".tfrecords"


if write_to_json:
    json_path = videoFile.split(".")[0] + "_tfrecords"
    os.mkdir(json_path)
    json_path += "/"

# initialize the list of reference points
refPt = []

def click_and_id(event, x, y, flags, param):

    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and draw a dot at the location.
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image, refPt[-1], 2, (255, 255, 255), 2)
        cv2.imshow('image', image)

    # if the left mouse button was released, record the ending (x,y)
    # coordinates and draw a rectangle with recorded starting and
    # ending coordinates.
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cv2.rectangle(image, refPt[-2], refPt[-1], (255, 255, 255), 2)
        cv2.imshow('image', image)

def order_bounding_box(corner1, corner2):
    '''
    Takes as input any two corners of a rectangle and returns two pairs of coordinates -
    upper left and lower right.
    '''
    upper_left = (min(corner1[0], corner2[0]), min(corner1[1], corner2[1]))
    lower_right = (max(corner1[0], corner2[0]), max(corner1[1], corner2[1]))
    return [upper_left, lower_right]

def update_dict_coords(output_dict, label, class_text):
    '''

    '''
    global refPt
    bb = order_bounding_box(refPt[-2], refPt[-1])
    output_dict['xmins'].append(bb[0][0])
    output_dict['ymins'].append(bb[0][1])
    output_dict['xmaxs'].append(bb[1][0])
    output_dict['ymaxs'].append(bb[1][1])
    output_dict['classes'].append(label)
    output_dict['classes_text'].append(class_text)

def create_tf_example(example):
    image_filename = str.encode(example['file_name'])
    height = example['image_height']
    width = example['image_width']
    image_format = b'jpg'
    encoded_image_data = example['image_encoded']

    xmins = list(np.array(example['xmins'])/width) # List of normalized left x coordinates in bounding box (1 per box)

    xmaxs = list(np.array(example['xmaxs'])/width) # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = list(np.array(example['ymins'])/height) # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = list(np.array(example['ymaxs'])/height) # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = []
    for label in example['classes_text']:
        classes_text.append(label.encode('utf-8')) # List of string class name of bounding box (1 per box)
    classes = example['classes'] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_filename),
        'image/source_id': dataset_util.bytes_feature(image_filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    }))
    return tf_example

if 'flags' not in vars():
    flags = tf.compat.v1.flags
    flags.DEFINE_string('output_path', '', "/Users/amandawhaley/Projects/UrbanVision/train.tfrecords")
    flags.DEFINE_string('f', '', 'kernel') #Had to add this workaround - not sure why
    FLAGS = flags.FLAGS
FLAGS.output_path = output_path

#Display a series of frames (with time gap between frames specified)
#from a video.  Tag the objects in each image.  Write data out to file.


writer = tf.io.TFRecordWriter(FLAGS.output_path)
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
frame_set = set(sample(range(int(video_duration*frameRate)), num_records))

records_count = 0
while(cap.isOpened() and records_count < num_records):
    frameId = cap.get(1) #current frame number
    ret, image = cap.read()
    if not ret:
        break
    if int(frameId) in frame_set:
        image = cv2.resize(image,None,fx=0.35,fy=0.35)
        file_name = 'frame_' + str(int(frameId)) + '.jpg'

        output_dict = {'file_name': file_name, 'image_width':None, 'image_height':None,
                       'xmins':[], 'xmaxs':[], 'ymins':[], 'ymaxs':[], 'classes':[],
                       'classes_text':[]}
        # Clone the image and setup the mouse callback function
        clone = image.copy()
        output_dict['image_height']=image.shape[0]
        output_dict['image_width']=image.shape[1]
        _, im_buf_arr = cv2.imencode(".jpg", image[:1,:1,:])
        cv2.namedWindow("image")
        cv2.startWindowThread()
        cv2.setMouseCallback("image", click_and_id)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()

            # if the 'p' key is pressed, print coordinates, 'person', and draw rectangle in green.
            if key == ord("p"):
                update_dict_coords(output_dict, 1, 'person')
                image = clone.copy()
                cv2.rectangle(image, refPt[-2], refPt[-1], (0, 255, 0), 3)
                cv2.imshow('image', image)
                clone = image.copy()
            # if the 'b' key is pressed, print coordinates, 'bus', and draw rectangle in red.
            if key == ord("b"):
                update_dict_coords(output_dict, 2, 'bus')
                image = clone.copy()
                cv2.rectangle(image, refPt[-2], refPt[-1], (0, 0, 255), 3)
                cv2.imshow('image', image)
                clone = image.copy()

            # if the 'c' key is pressed, print coordinates, 'car', and draw rectangle in yellow.
            if key == ord("c"):
                update_dict_coords(output_dict, 3, 'car')
                image = clone.copy()
                cv2.rectangle(image, refPt[-2], refPt[-1], (0, 255, 255), 3)
                cv2.imshow('image', image)
                clone = image.copy()

            # if the 'q' key is pressed, break from the loop
            elif key == ord("q"):
                break

        #Write output_dict to file
        if output_dict['classes'] and write_to_json:
            file_name_json = 'train_data_frame_' + str(int(frameId)) + '.json'
            with open(json_path + file_name_json, 'w') as fp:
                json.dump(output_dict, fp)

        #Write tagged image to file
        if output_dict['classes'] and save_tagged_image:
            file_name_jpg = 'tagged_image_frame_' + str(int(frameId)) + '.jpg'
            cv2.imwrite(file_name_jpg,image)

        # close all open windows
        cv2.destroyAllWindows()
        cv2.waitKey(10)
        records_count += 1
        output_dict['image_encoded']=im_buf_arr.tobytes()
        tf_example = create_tf_example(output_dict)
        writer.write(tf_example.SerializeToString())

cap.release()
cv2.waitKey(10)
writer.close()

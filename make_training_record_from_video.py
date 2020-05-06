import cv2
import math
import json
import time
import warnings
import tensorflow as tf
import numpy as np
from object_detection.utils import dataset_util
import os, glob
from PIL import Image
import pathlib
from random import sample
import argparse
import train


def order_bounding_box(corner1, corner2):
    '''
    Takes as input any two corners of a rectangle and returns two pairs of coordinates -
    upper left and lower right.
    '''
    upper_left = (min(corner1[0], corner2[0]), min(corner1[1], corner2[1]))
    lower_right = (max(corner1[0], corner2[0]), max(corner1[1], corner2[1]))
    return [upper_left, lower_right]

def update_dict_coords(output_dict, label, class_text, corner1, corner2):
    '''

    '''
    bb = order_bounding_box(corner1, corner2)
    output_dict['xmins'].append(bb[0][0])
    output_dict['ymins'].append(bb[0][1])
    output_dict['xmaxs'].append(bb[1][0])
    output_dict['ymaxs'].append(bb[1][1])
    output_dict['classes'].append(label)
    output_dict['classes_text'].append(class_text)

def main():

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path of the video file.')
    parser.add_argument('--records', default=20, help='Number of frames to tag.', type=int)
    parser.add_argument('--duration', help='Duration of the video in minutes.', required=True, type=float)
    parser.add_argument('--width', help='Screen width in pixels.', type = int, required=True)
    parser.add_argument('--height', help='Screen height in pixels.', type = int, required=True)
    parser.add_argument("--save", default = False, help='Boolean: save tagged images')
    parser.add_argument("--json", default = False, help='Boolean: Write results out to json')

    args = parser.parse_args()

    num_records = args.records
    save_tagged_image = args.save
    write_to_json = args.json
    videoFile = args.video
    print("args.duration type: ", type(args.duration))
    video_duration = args.duration*60.
    screen_width = args.width
    screen_height = args.height

    screen_dim = (screen_width, screen_height)

    train.print_tagging_instructions()

    frames_tagged, output_path = train.get_priors(videoFile)
    print("frames_tagged: ", frames_tagged)
    print("output_path: ", output_path)

    if write_to_json:
        json_path = videoFile.split(".")[0] + "_tfrecords"
        os.mkdir(json_path)
        json_path += "/"

    #Display a series of frames (with time gap between frames specified)
    #from a video.  Tag the objects in each image.  Write data out to file.

    writer = tf.io.TFRecordWriter(output_path)
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5) #frame rate
    frame_set = set(sample(range(int(video_duration*frameRate)), num_records))

    records_count = 0
    time_tagging = 0
    try:
        while(cap.isOpened() and records_count < num_records):
            frameId = cap.get(1) #current frame number
            ret, image = cap.read()
            if not ret:
                break
            if int(frameId) in frame_set and int(frameId) not in frames_tagged:
                print("Frame number {}.  This is record {} out of {}.".format(
                                frameId, records_count + 1, num_records))
                image = cv2.resize(image, screen_dim, interpolation = cv2.INTER_AREA)
                source_id = 'frame_' + str(int(frameId))

                output_dict = {'source_id': source_id, 'image_width':None, 'image_height':None,
                                'xmins':[], 'xmaxs':[], 'ymins':[], 'ymaxs':[], 'classes':[],
                                'classes_text':[]}
                # Clone the image and setup the mouse callback function
                clone = image.copy()
                output_dict['image_height']=image.shape[0]
                output_dict['image_width']=image.shape[1]
                _, im_buf_arr = cv2.imencode(".jpg", image[:1,:1,:])
                cv2.namedWindow("image")
                cv2.startWindowThread()
                t0= time.clock()

                # keep looping until the 'q' key is pressed
                while True:
                    # display the image and wait for a keypress
                    cv2.imshow("image", image)
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

                    # if the 'p' key is pressed, print coordinates, 'person', and draw rectangle in green.
                    if key == ord("p"):
                        image = clone.copy()
                        cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 255, 0), 1)
                        update_dict_coords(output_dict, 1, 'person', corner1, corner2)
                        cv2.imshow('image', image)
                        clone = image.copy()

                    # if the 'b' key is pressed, print coordinates, 'bus', and draw rectangle in red.
                    if key == ord("b"):
                        image = clone.copy()
                        cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 0, 255), 1)
                        update_dict_coords(output_dict, 2, 'bus', corner1, corner2)
                        cv2.imshow('image', image)
                        clone = image.copy()

                    # if the 'c' key is pressed, print coordinates, 'car', and draw rectangle in yellow.
                    if key == ord("c"):
                        image = clone.copy()
                        cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 255, 255), 1)
                        update_dict_coords(output_dict, 3, 'car', corner1, corner2)
                        cv2.imshow('image', image)
                        clone = image.copy()

                    # if the 'q' key is pressed, break from the loop
                    elif key == ord("q"):
                        frame_time = time.clock() - t0
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
                output_dict['image_encoded']=im_buf_arr.tobytes()
                tf_example = train.create_tf_example(output_dict)
                writer.write(tf_example.SerializeToString())
                records_count += 1
                time_tagging += frame_time
                print("Running tag rate: {} seconds per frame".format(time_tagging/records_count))

    except KeyboardInterrupt:
        print("Tagging interrupted.  Writing out {} records instead of {}.".format(
                records_count, num_records))

    cap.release()
    cv2.waitKey(10)
    writer.close()
    if not records_count:
        os.remove(output_path)
        print("No frames tagged.")
    else:
        train.print_summary(records_count, output_path)

if __name__ == '__main__': main()

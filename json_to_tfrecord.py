import argparse
import json
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
from random import sample
from random import randint

import train




parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, help='Path of the json file.')
parser.add_argument('--video', required=True, help='Path of the video file.')
parser.add_argument('--val', default = 0.2, help='Fraction of frames for validation set',
                    type = float)
parser.add_argument('--slim', default = 10, help='Maximum number of sequential frames.')


args = parser.parse_args()

videoFile = args.video
videoFile_pathless = videoFile.split("/")[-1]
print(videoFile_pathless)
json_filename = args.file
print(args.slim)
# Read in json as dictionary.
fp = open(json_filename, "r")
tagged_dict = json.load(fp)
fp.close()


# Get a sorted list of frame numbers.
frame_set = set(map(int, tagged_dict.keys()))

# Look for tfrecords for this video that already exist.  Make tfrecord output path.
vol, frames_tagged, train_output_path = train.get_priors(videoFile_pathless)
print("{} frames already tagged from this video.".format(len(frames_tagged)))
print("output_paths: ")
print(train_output_path)
val_output_path = train_output_path.replace('_train', '_val')
print(val_output_path)

#Initialize writer
train_writer = tf.io.TFRecordWriter(train_output_path)
val_writer = tf.io.TFRecordWriter(val_output_path)

# Iterate through list of frames and create a tfrecord.
cap = cv2.VideoCapture(videoFile)
last_frame = -2

train_count = 0
val_count = 0
in_a_row = 1
while(cap.isOpened()):
    frameId = int(cap.get(1)) #current frame number
    ret, image = cap.read()
    if not ret:
        print("Did not read frame")
        break

    #Check if current frame is in the frame set and is not a duplicate.
    if frameId in frame_set and frameId not in frames_tagged:
        if frameId - last_frame > 1:
            in_a_row = 1
        else:
            in_a_row += 1
        if not args.slim or in_a_row <= args.slim:
            output_dict = tagged_dict[str(frameId)]
            _, im_buf_arr = cv2.imencode(".jpg", image)
            output_dict['image_encoded']=im_buf_arr.tobytes()

            #Write out to tfrecord
            tf_example = train.create_tf_example(output_dict)

            if frameId - last_frame > 1:
                x = np.random.uniform()

            if x < args.val:
                val_writer.write(tf_example.SerializeToString())
                val_count += 1
            else:
                train_writer.write(tf_example.SerializeToString())
                train_count += 1
        last_frame = frameId

train_writer.close()
val_writer.close()
print("*******************************************************************")
print("Summary:")
print("{} tfrecord were saved.".format(train_count + val_count))
print("{} records in {}, and {} records in {}.".format(train_count,
                                                    train_output_path,
                                                    val_count,
                                                    val_output_path))
print("*******************************************************************")

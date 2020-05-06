import cv2
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from random import sample
from random import randint
import argparse

import train


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path of the video file.')
    parser.add_argument('--records', default=20, help='Number of frames to tag.',
                        type=int)
    parser.add_argument('--duration', help='Duration of the video in minutes.',
                        required=True, type=float)
    parser.add_argument('--width', help='Screen width in pixels.', type = int,
                        required=True)
    parser.add_argument('--height', help='Screen height in pixels.', type = int,
                        required=True)
    parser.add_argument('--val', default = 0.2, help='Fraction of frames for validation set',
                        type = float)
    parser.add_argument("--save", default = False, help='Boolean: save tagged images')
    parser.add_argument("--json", default = False, help='Boolean: Write results out to json')

    args = parser.parse_args()

    num_records = args.records
    save_tagged_image = args.save
    write_to_json = args.json
    videoFile = args.video
    video_duration = args.duration*60.
    screen_width = args.width
    screen_height = args.height

    screen_dim = (screen_width, screen_height)

    # Specify the tracker type
    trackerType = "CSRT"

    #Get prior frames
    #Make frame set
    frames_tagged, train_output_path = train.get_priors(videoFile)
    print("{} frames already tagged from this video.".format(len(frames_tagged)))
    print("frames_tagged: ", len(frames_tagged))
    print("output_paths: ")
    print(train_output_path)
    val_output_path = train_output_path.replace('train_', 'val_')
    print(val_output_path)

    #Make frames to tag list
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5) #frame rate
    try:
        frame_set = set(sample(range(int(video_duration*frameRate)), num_records))
    except ValueError:
        print("Sample larger than population or is negative - \
                Choose a different number of frames to tag.")

    #Initialize writer
    train_writer = tf.io.TFRecordWriter(train_output_path)
    val_writer = tf.io.TFRecordWriter(val_output_path)

    records_count = 0
    num_tagged = 0
    time_tagging = 0
    frameId = -1
    train_count = 0
    val_count = 0

    #Iterate through frames in video - tag and track frames listed in the
    #frame set.  A keyboard interrupt will result in writing out to file before
    #all frames in frame set have been tagged.
    try:
        while(cap.isOpened() and num_tagged < num_records):
            frameId = cap.get(1) #current frame number
            ret, image = cap.read()
            if not ret:
                print("Did not read frame")
                break
            if frameId % 250 == 0: print("Frame number {}.".format(int(frameId)))
            #Check if current frame is in the frame set and is not a duplicate.
            if int(frameId) in frame_set and int(frameId) not in frames_tagged:
                print("Frame number {}.  This is record {} out of {}.".format(
                                int(frameId), num_tagged + 1, num_records))

                #Tag all objects in frame - save info as output_dict
                image = cv2.resize(image, screen_dim, interpolation = cv2.INTER_AREA)
                output_dict, frame_time = train.tag_objects(frameId, image, videoFile)

                #Write out to tfrecord
                tf_example = train.create_tf_example(output_dict)
                x = np.random.uniform()
                if x < args.val:
                    val_writer.write(tf_example.SerializeToString())
                    val_count += 1
                else:
                    train_writer.write(tf_example.SerializeToString())
                    train_count += 1
                records_count += 1
                num_tagged += 1
                time_tagging += frame_time
                print("Running tag rate: {} seconds per frame".format(
                        time_tagging/records_count))

                #Set tracker bboxes as list of tagged objects
                bboxes = []
                colors = []
                for i in range(len(output_dict['classes'])):
                    bboxes.append((output_dict['xmins'][i],
                                   output_dict['ymins'][i],
                                   output_dict['xmaxs'][i] - output_dict['xmins'][i],
                                   output_dict['ymaxs'][i] - output_dict['ymins'][i]))
                    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

                #Create new multitracker object
                multiTracker = cv2.MultiTracker_create()

                #Add bboxes to tracker object
                for bbox in bboxes:
                    multiTracker.add(train.createTrackerByName(trackerType), image, bbox)

                #Read next frame and update multitracker object
                #Display bboxes (on cloned image) and ask user if they accept the
                #frame or not.  If not quit
                while True:
                    next_frameId = cap.get(1)
                    ret, next_image = cap.read()
                    next_image = cv2.resize(next_image, screen_dim,
                                            interpolation = cv2.INTER_AREA)
                    clone = next_image.copy()
                    if not ret:
                        print("Could not read frame.")
                        break
                    ret, boxes = multiTracker.update(next_image)

                    # draw tracked objects
                    for i, newbox in enumerate(boxes):
                        p1 = (int(newbox[0]), int(newbox[1]))
                        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                        cv2.rectangle(clone, p1, p2, colors[i], 2, 1)

                    # show frame
                    cv2.namedWindow("MultiTracker")
                    cv2.startWindowThread()
                    cv2.imshow('MultiTracker', clone)
                    cv2.waitKey(100)
                    print("Press y to accept this tagged frame.")
                    print("Press any other key to quit without accepting.")
                    k = cv2.waitKey(0) & 0xFF
                    if (k == 121):  # y is pressed
                        next_output_dict = train.update_tracked(output_dict,
                                                                boxes,
                                                                next_image,
                                                                next_frameId)
                        next_tf_example = train.create_tf_example(next_output_dict)
                        x = np.random.uniform()
                        if x < args.val:
                            val_writer.write(next_tf_example.SerializeToString())
                            val_count += 1
                        else:
                            train_writer.write(next_tf_example.SerializeToString())
                            train_count += 1
                        records_count += 1
                    else:
                        cv2.destroyWindow('MultiTracker')
                        cv2.waitKey(100)
                        break

    except KeyboardInterrupt:
        print("Tagging interrupted.  Writing out {} records instead of {}.".format(
                num_tagged, num_records))

    cap.release()
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    cv2.waitKey(10)
    train_writer.close()
    val_writer.close()
    train.print_summary(train_count, val_count, train_output_path, val_output_path)

if __name__ == '__main__': main()

import csv
import argparse
import os
import cv2
from random import sample
from random import randint
import json
import tag

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True, help='Path of the video file.')
parser.add_argument('--records', default=20, help='Number of frames to tag.',
                        type=int)
parser.add_argument('--duration', help='Duration of the video in minutes.',
                        required=True, type=float)

args = parser.parse_args()

videoFile = args.video
num_records = args.records
video_duration = args.duration*60.

# Specify the tracker type
trackerType = "CSRT"


stem = videoFile.split(".")[0]
json_filename = stem + "_records.json"

#Add code to read in frames from csv.

#Check if json_filename exists.
if not os.path.exists(json_filename):
    print("No json files exist for this video.  Output path: ", json_filename)
    tagged_dict = {}
    frames_tagged = set()
elif not os.path.getsize(json_filename):
    print("The json file {} is empty.  Writing out to file.".format(json_filename))
    tagged_dict = {}
    frames_tagged = set()
else:
    fp = open(json_filename, "r")
    tagged_dict = json.load(fp)
    fp.close()
    frames_tagged = set(tagged_dict.keys())
    print("{} already exists with {} frames tagged.  Appending to file.".format(
                json_filename, len(frames_tagged)))

tag.print_tagging_instructions()

#Make frames to tag list
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
try:
    frame_set = set(sample(range(int(video_duration*frameRate)), num_records))
except ValueError:
    print("Sample larger than population or is negative - \
            Choose a different number of frames to tag.")

records_count = 0
num_tagged = 0
frameId = -1

try:
    while(cap.isOpened() and num_tagged < num_records):
        frameId = int(cap.get(1)) #current frame number
        ret, image = cap.read()
        if not ret:
            print("Did not read frame")
            break

        if frameId % 250 == 0: print("Frame number {}.".format(frameId))
        #Check if current frame is in the frame set and is not a duplicate.
        if frameId in frame_set and frameId not in frames_tagged:
            print("Frame number {}.  This is record {} out of {}.".format(
                        frameId, num_tagged + 1, num_records))

            output_dict = tag.tag_objects(frameId, image, videoFile)

            #Write output_dict to file
            tagged_dict[frameId] = output_dict
            with open(json_filename, 'w') as fp:
                json.dump(tagged_dict, fp)
            fp.close()

            records_count += 1
            num_tagged += 1
            frames_tagged.add(frameId)

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
                multiTracker.add(tag.createTrackerByName(trackerType), image, bbox)

            #Read next frame and update multitracker object
            #Display bboxes (on cloned image) and ask user if they accept the
            #frame or not.  If not quit
            while True:
                next_frameId = int(cap.get(1))
                ret, next_image = cap.read()
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
                    next_output_dict = tag.update_tracked(output_dict,
                                                        boxes,
                                                        next_image,
                                                        next_frameId)
                    #Write output_dict to file
                    tagged_dict[next_frameId] = next_output_dict
                    with open(json_filename, 'w') as fp:
                        json.dump(tagged_dict, fp)
                    fp.close()
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


tag.print_summary(num_tagged, json_filename)

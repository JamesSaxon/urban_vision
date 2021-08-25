import os, glob, time
import tensorflow as tf
import numpy as np
import cv2


def print_tagging_instructions():
    '''
    Prints instructions for tagging objects.
    '''
    print("*******************************************************************")
    print("Instructions for tagging frames")
    print("1. Drag a bounding box around the visible part of the object.")
    print('2. Press "enter" to accept bounding box.  Press "c" to redo.  \
            Bounding box should turn white when accepted.')
    print("3.  Once the bounding box is white, press 'p' for person, 'c' for car,\
            'b' for bus, 'r' to redo, and 'q' to quit the frame.")
    print("4.  If you tagged a person, car, or bus, the bounding box should change color.")
    print("5.  To quit before the number of frames specified has been tagged, press Ctrl-C.")
    print("6.  Tag every object (person, car, or bus) in the frame")
    print("7.  Make sure bounding box covers only the visible parts of the object.")
    print("8.  Don't tag objects that are mostly hidden.")
    print("*******************************************************************")

def print_summary(train_count, val_count, train_output_path, val_output_path):
    '''
    Prints a summary once tagging is finished.
    '''
    train_filename = train_output_path.split('/')[-1]
    val_filename = val_output_path.split('/')[-1]
    print("*******************************************************************")
    print("Summary:")
    print("You tagged {} frames.".format(train_count + val_count))
    print("{} records in {}, and {} records in {}.".format(train_count,
                                                        train_filename,
                                                        val_count,
                                                        val_filename))
    print("*******************************************************************")

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(example):
    '''
    Converts a dictionary of features for a single frame to a tf_example object.
    '''
    video = str.encode(example['video'])
    source_id = str.encode(example['source_id'])
    height = example['image_height']
    width = example['image_width']
    image_format = str.encode('jpg')
    encoded_image_data = example['image_encoded']

    xmins = list(np.array(example['xmins'])/width)  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = list(np.array(example['xmaxs'])/width)  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = list(np.array(example['ymins'])/height) # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = list(np.array(example['ymaxs'])/height) # List of normalized bottom y coordinates in bounding box (1 per box)

    classes_text = []
    for label in example['classes_text']:
        classes_text.append(label.encode('utf-8')) # List of string class name of bounding box (1 per box)
    classes = example['classes'] # List of integer class id of bounding box (1 per box)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/video': _bytes_feature(video),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/source_id': _bytes_feature(source_id),
        'image/encoded': _bytes_feature(encoded_image_data),
        'image/format': _bytes_feature(image_format),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/label': _int64_list_feature(classes),
        'image/object/class/text': _bytes_list_feature(classes_text),
    }))
    return tf_example

def read_tfrecord(serialized_example):
    '''
    Takes a serialized example (from a tfrecord file) and returns a dictionary
    with the same features.
    '''
    feature_description = {
        'image/video': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'image/object/class/label': tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
        'image/object/class/text': tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    return example


def read_tfrecord_framenumber(serialized_example):
    '''
    Takes a serialized example (from a tfrecord file) and returns the frame
    number feature as a tf tensor.
    '''
    feature_description = {
        'image/video': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'image/object/class/label': tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
        'image/object/class/text': tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    return example['image/source_id']


def get_priors(videoFile):
    '''
    Looks for tfrecord files in the same directory that contain records from
    the specified video and returns a list of the frames that have already
    been tagged from that video.

    Also returns an available output path for a new tfrecord file.
    '''
    stem_train = videoFile.split(".")[0] + "_train.record"
    stem_val = videoFile.split(".")[0] + "_val.record"
    record_group = glob.glob(stem_train + '*')
    record_group.extend(glob.glob(stem_val + '*'))
    output_path = stem_train + "-00000-of-00001"
    if not record_group:
        return 1, set(), output_path

    #If prior records for this video, get volume numbers:
    volumes = []
    file_paths = []
    for record in record_group:
        volumes.append(int(record.split("-")[-3]))
        if os.path.getsize(record):
            file_paths.append(record)

    frames_tagged = set()
    if file_paths:
        tfrecord_dataset = tf.data.TFRecordDataset(file_paths)
        parsed_dataset = tfrecord_dataset.map(read_tfrecord_framenumber)
        for frame_bytestring in parsed_dataset:
            frames_tagged.add(int(str(frame_bytestring.numpy(), 'utf-8')))
    volume = max(volumes) + 1
    output_path = stem_train + "-" + str(volume).zfill(5) + "-of-" + str(volume+1).zfill(5)
    return volume+1, frames_tagged, output_path


def update_record_names(output_path, vol):
    '''
    Changes the volume number in file names of train and val records with the same
    starting path.
    '''
    video_id = output_path.split("_train.record")[0]
    for filepath in glob.glob(video_id + "_*.record*"):
        filename_list = filepath.split("-")
        filename_list[-1] = str(vol).zfill(5)
        new_filepath = "-".join(filename_list)
        os.replace(filepath, new_filepath)




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


def tag_objects(frameId, image, videoFile, write_to_json=False, save_tagged_image=False):
    '''
    Initializes and completes the tagging process for a single frame.
    Inputs:
        frameId - (string) frame number
        image - numpy array
        videoFile - name of the video from which the frame was taken
    Returns:
        output_dict - dictionary of features which can then be serialized to a
                      tf_example object
        frame_time - (int) time in seconds it took the user to tag the frame.
    '''
    output_dict = {'video': str(videoFile), 'source_id': str(int(frameId)), 'image_width':None, 'image_height':None,
                    'xmins':[], 'xmaxs':[], 'ymins':[], 'ymaxs':[], 'classes':[],
                    'classes_text':[]}
    output_dict['image_height']=image.shape[0]
    output_dict['image_width']=image.shape[1]
    _, im_buf_arr = cv2.imencode(".jpg", image)
    output_dict['image_encoded']=im_buf_arr.tobytes()
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.startWindowThread()
    t0= time.clock()

    while True:
        # display the image and allow bounding box selection
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
            frame_time = time.clock() - t0
            break

    #Print tagging time

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

    return output_dict, frame_time

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
    new_output_dict['source_id'] = str(int(next_frameId))

    _, next_im_buf_arr = cv2.imencode(".jpg", clone)
    new_output_dict['image_encoded']=next_im_buf_arr.tobytes()

    return new_output_dict
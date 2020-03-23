import os
import tensorflow as tf
import numpy as np
from object_detection.utils import dataset_util


def create_tf_example(example):
    source_id = str.encode(example['source_id'])
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
        'image/source_id': dataset_util.bytes_feature(source_id),
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

def read_tfrecord(serialized_example):
    feature_description = {
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
    image_height = example['image/height']
    image_width = example['image/width']
    image_source_id = example['image/source_id']
    image_encoded = example['image/encoded']
    image_format = example['image/format']
    image_xmins = example['image/object/bbox/xmin']
    image_xmaxs = example['image/object/bbox/xmax']
    image_ymins = example['image/object/bbox/ymin']
    image_ymaxs = example['image/object/bbox/ymax']
    image_labels = example['image/object/class/label']
    image_classes = example['image/object/class/text']

    return example


def read_tfrecord_framenumber(serialized_example):
    feature_description = {
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


def get_frames_tagged(records_file):
    

def get_priors(videoFile):
    output_path = os.getcwd() + "/train_" + \
                    videoFile.split(".")[0] + "_1.tfrecords"

    frames_tagged = set()
    file_paths = []
    while os.path.exists(output_path):
        print("{} already exists.".format(output_path))
        vol = int(output_path.split(".")[0].split("_")[-1])
        file_paths.append("train_" + videoFile.split(".")[0] + "_" + \
                          str(vol) + ".tfrecords")
        vol += 1
        output_path = os.getcwd() + "/train_" + \
                        videoFile.split(".")[0] + "_" + str(vol) + ".tfrecords"
        print("new output path: ", output_path)

    if file_paths:
        print("file_paths: ", file_paths)
        tfrecord_dataset = tf.data.TFRecordDataset(file_paths)
        parsed_dataset = tfrecord_dataset.map(read_tfrecord_framenumber)
        for frame_string in parsed_dataset:
            frames_tagged.add(int(str(frame_string.numpy())[8:-1]))

    return frames_tagged, output_path

#!/usr/bin/env python

import cv2 
import os, sys, re
from time import sleep
from copy import deepcopy as dc
from glob import glob

import numpy as np
import tensorflow as tf

import random

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', "--ifile",  type = str, required = True)
parser.add_argument('-o', "--ofile",  type = str, required = True)
parser.add_argument("-c", "--categs", default = ["car", "truck", "bus"],
                    choices = ["person", "bike", "car", "truck", "bus"], 
                    nargs = "+", help = "Types of objects to process -- must match labels")
parser.add_argument('-s', "--min_size", type = float, default = 0)

args = parser.parse_args()

CATEGS = args.categs
SIZE   = args.min_size

image_feature_description = {
   'image/video'              : tf.io.FixedLenFeature((), tf.string),
   'image/height'             : tf.io.FixedLenFeature((), tf.int64),
   'image/width'              : tf.io.FixedLenFeature((), tf.int64),
   'image/tag'                : tf.io.FixedLenFeature((), tf.string),
   'image/timestamp'          : tf.io.FixedLenFeature((), tf.string),
   'image/frame_id'           : tf.io.FixedLenFeature((), tf.int64),
   'image/encoded'            : tf.io.FixedLenFeature((), tf.string),
   'image/format'             : tf.io.FixedLenFeature((), tf.string),
   'image/object/bbox/xmin'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/bbox/xmax'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/bbox/ymin'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/bbox/ymax'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/bbox/conf'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/class/label' : tf.io.FixedLenSequenceFeature([], dtype = tf.int64  , allow_missing = True),
   'image/object/class/text'  : tf.io.FixedLenSequenceFeature([], dtype = tf.string , allow_missing = True),
}


def _parse_image_function(protob):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(protob, image_feature_description)

def _int64_feature(value):      return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):      return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def _int64_list_feature(value): return tf.train.Feature(int64_list = tf.train.Int64List(value = list(value)))
def _bytes_list_feature(value): return tf.train.Feature(bytes_list = tf.train.BytesList(value = list(value)))
def _float_list_feature(value): return tf.train.Feature(float_list = tf.train.FloatList(value = list(value)))


label_dict = {c : ci+1 for ci, c in enumerate(CATEGS)}
cat_dict = {v : k for k, v in label_dict.items()}

def make_record(original, detection_dict):
    
    d = detection_dict

    int_labels = [label_dict[txt] for txt in d["label"]]
    txt_labels = [txt.encode('utf-8') for txt in d["label"]]

    return tf.train.Example(features = tf.train.Features(feature={
        'image/video'              : _bytes_feature(original['image/video'].numpy()),
        'image/frame_id'           : _int64_feature(original['image/frame_id'].numpy()),
        'image/tag'                : _bytes_feature(original['image/tag'].numpy()),
        'image/timestamp'          : _bytes_feature(original['image/timestamp'].numpy()),
        'image/encoded'            : _bytes_feature(original["image/encoded"].numpy()),
        'image/format'             : _bytes_feature(original["image/format"].numpy()),

        'image/sanitized'          : _int64_feature(False),

        'image/height'             : _int64_feature(d["height"]),
        'image/width'              : _int64_feature(d["width"]),
        'image/object/bbox/xmin'   : _float_list_feature(d["xmin"]),
        'image/object/bbox/xmax'   : _float_list_feature(d["xmax"]),
        'image/object/bbox/ymin'   : _float_list_feature(d["ymin"]),
        'image/object/bbox/ymax'   : _float_list_feature(d["ymax"]),
        'image/object/bbox/conf'   : _float_list_feature(d["conf"]),


        'image/object/class/label' : _int64_list_feature(int_labels),
        'image/object/class/text'  : _bytes_list_feature(txt_labels),
    }))

def get_detections_dict(detections):

    width  = detections['image/width'].numpy()
    height = detections['image/height'].numpy()

    return {
      "width"  : width,
      "height" : height,
      "label"  : detections['image/object/class/text'].numpy().astype(str),
      "xmin"   : detections['image/object/bbox/xmin'].numpy().astype(float),
      "xmax"   : detections['image/object/bbox/xmax'].numpy().astype(float),
      "ymin"   : detections['image/object/bbox/ymin'].numpy().astype(float),
      "ymax"   : detections['image/object/bbox/ymax'].numpy().astype(float),
      "conf"   : detections['image/object/bbox/conf'].numpy().astype(float)
    }

def filter_detections(detections, size):

    d = detections 
    od = {"width" : d["width"], "height" : d["height"], 
          "xmin" : [], "xmax" : [], "ymin" : [], "ymax" : [], "conf" : [], "label" : []}

    for xmin, xmax, ymin, ymax, conf, label in \
        zip(d["xmin"], d["xmax"], d["ymin"], d["ymax"], d["conf"], d["label"]):

        if size * size > (xmax - xmin) * (ymax - ymin): continue

        od["xmin"].append(xmin)
        od["xmax"].append(xmax)
        od["ymin"].append(ymin)
        od["ymax"].append(ymax)
        od["conf"].append(conf)
        od["label"].append(label)

    for v in ["xmin", "xmax", "ymin", "ymax", "conf", "label"]:

        od[v] = np.array(od[v])
        if v[0] in "xy":
            od[v] = od[v].astype(float)

    return od




tfrecord_dataset = tf.data.TFRecordDataset(glob(args.ifile))
parsed_dataset = tfrecord_dataset.map(_parse_image_function)
parsed_list = list(parsed_dataset)
random.shuffle(parsed_list)

output_records = tf.io.TFRecordWriter(args.ofile)

for ix, record in enumerate(parsed_list):

    detections = get_detections_dict(record)

    odetections = filter_detections(detections, SIZE)

    orecord = make_record(record, odetections)

    output_records.write(orecord.SerializeToString())

output_records.close()

print()
print(ix+1, "records")
print(label_dict)


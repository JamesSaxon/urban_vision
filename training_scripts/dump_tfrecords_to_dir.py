#!/usr/bin/env python 

import cv2
import tensorflow as tf
import numpy as np
import argparse
import os

import random

from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--tfrecords", required=True, help='Path of the tfrecord file.')
parser.add_argument("-d", "--directory", required=True, help="Output directory")
parser.add_argument("--val_frac", type = int, default = 10, help="1 in how many are validation?")
args = parser.parse_args()

VFRAC = args.val_frac

RECORD_PATH = args.tfrecords
OUTPUT_PATH = args.directory
if OUTPUT_PATH[-1] == "/":
    OUTPUT_PATH = OUTPUT_PATH[:-1]

print(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH, exist_ok = True)
for tag in ["TEST", "TRAIN"]:
    os.makedirs(f"{OUTPUT_PATH}/{tag}",            exist_ok = True)
    os.makedirs(f"{OUTPUT_PATH}/{tag}/labels",     exist_ok = True)
    os.makedirs(f"{OUTPUT_PATH}/{tag}/gt",         exist_ok = True)
    os.makedirs(f"{OUTPUT_PATH}/{tag}/JPEGImages", exist_ok = True)


label_dict = {"person" : 0, "bicycle" : 0, "car" : 0, "truck" : 1, "bus" : 2}

def _int64_feature(value):      return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):      return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def _int64_list_feature(value): return tf.train.Feature(int64_list = tf.train.Int64List(value =  value ))
def _bytes_list_feature(value): return tf.train.Feature(bytes_list = tf.train.BytesList(value =  value ))
def _float_list_feature(value): return tf.train.Feature(float_list = tf.train.FloatList(value =  value ))


image_feature_description = {
   'image/video'              : tf.io.FixedLenFeature((), tf.string),
   'image/height'             : tf.io.FixedLenFeature((), tf.int64),
   'image/width'              : tf.io.FixedLenFeature((), tf.int64),
   'image/tag'                : tf.io.FixedLenFeature((), tf.string),
   'image/timestamp'          : tf.io.FixedLenFeature((), tf.string),
   'image/frame_id'           : tf.io.FixedLenFeature((), tf.int64),
   'image/encoded'            : tf.io.FixedLenFeature((), tf.string),
   'image/format'             : tf.io.FixedLenFeature((), tf.string),
   'image/sanitized'          : tf.io.FixedLenFeature((), tf.int64),
   'image/object/bbox/xmin'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/bbox/xmax'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/bbox/ymin'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/bbox/ymax'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/bbox/conf'   : tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True),
   'image/object/class/label' : tf.io.FixedLenSequenceFeature([], dtype = tf.int64  , allow_missing = True),
   'image/object/class/text'  : tf.io.FixedLenSequenceFeature([], dtype = tf.string , allow_missing = True),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

tfrecord_dataset = tf.data.TFRecordDataset([RECORD_PATH])
parsed_dataset = list(tfrecord_dataset.map(_parse_image_function))
# random.shuffle(parsed_dataset)

total_records = len(parsed_dataset)

ml_csv_out = open(f"{OUTPUT_PATH}/ml.csv", "w")

n_train, n_test = 0, 0
for ix, frame in tqdm(enumerate(parsed_dataset), total = total_records):

    image_raw = frame['image/encoded'].numpy()
    image_encoded = np.frombuffer(image_raw, np.uint8)
    image = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)

    LABEL = frame['image/object/class/text'].numpy().astype(str)
    XMIN  = frame['image/object/bbox/xmin'].numpy().astype(float)
    XMAX  = frame['image/object/bbox/xmax'].numpy().astype(float)
    YMIN  = frame['image/object/bbox/ymin'].numpy().astype(float)
    YMAX  = frame['image/object/bbox/ymax'].numpy().astype(float)

    if not frame['image/sanitized'].numpy(): break

    tag = "TRAIN" if (ix % VFRAC) else "TEST"

    if tag == "TRAIN":
        n_img = n_train
        n_train += 1

    else:
        n_img = n_test
        n_test += 1


    cv2.imwrite(f"{OUTPUT_PATH}/JPEGImages/{n_img:05d}.jpg", image)
    cv2.imwrite(f"{OUTPUT_PATH}/{tag}/JPEGImages/{n_img:05d}.jpg", image)

    label_out = open(f"{OUTPUT_PATH}/{tag}/labels/{n_img:05d}.txt", "w")
    gt_out    = open(f"{OUTPUT_PATH}/{tag}/gt/{n_img:05d}.txt", "w")

    for label, xmin, xmax, ymin, ymax in \
        zip(LABEL, XMIN, XMAX, YMIN, YMAX):

        ncat = label_dict[label]
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        label_out.write(f"{ncat} {x:.3f} {y:.3f} {w:.3f} {h:.3f}\n")
        gt_out   .write(f"{label} {x:.3f} {y:.3f} {w:.3f} {h:.3f}\n")

    label_out.close()
    gt_out.close()

    for label, xmin, xmax, ymin, ymax in \
        zip(LABEL, XMIN, XMAX, YMIN, YMAX):

        ml_csv_out.write(f"{tag},CONFDIR/{OUTPUT_PATH}/{tag}/JPEGImages/{n_img:05d}.jpg,{label},{xmin},{ymin},,,{xmax},{ymax},,\n")


ml_csv_out.close()



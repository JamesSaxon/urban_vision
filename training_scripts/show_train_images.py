#!/usr/bin/env python 

import cv2
import tensorflow as tf
import numpy as np
import argparse

import train

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

parser = argparse.ArgumentParser()
parser.add_argument('--file',  required=True, help='Path of the tfrecord file.')
parser.add_argument('--scale', type = float, default = 1.0)
parser.add_argument('--limit', help='Number of frames to show', type = int)

args = parser.parse_args()
file_path = args.file

if args.limit:
    limit = args.limit

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

tfrecord_dataset = tf.data.TFRecordDataset([file_path])
parsed_dataset = tfrecord_dataset.map(_parse_image_function)
cv2.namedWindow("image")
cv2.startWindowThread()
ct = 0

colors = {"person"  : (0, 0, 255),
          "car"     : (255, 0, 0),
          "truck"   : (0, 255, 255),
          "bus"     : (0, 255, 0), 
          "bicycle" : (255, 0, 255)}


total_frames = len(list(parsed_dataset))
for ix, example in enumerate(parsed_dataset):

    image_raw = example['image/encoded'].numpy()
    image_encoded = np.frombuffer(image_raw, np.uint8)
    image = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)
    image_width = example['image/width'].numpy()
    image_height = example['image/height'].numpy()
    labels = example['image/object/class/text'].numpy().astype(str)
    label_nums = example['image/object/class/label'].numpy()

    tag = example['image/tag'].numpy().decode("utf-8")
    timestamp = example['image/timestamp'].numpy().decode("utf-8")
    frame = example['image/frame_id'].numpy()

    print("{}/{} #{} ({}/{})".format(tag, timestamp, frame, ix, total_frames))

    for i, label in enumerate(labels):

        xmin = int(example['image/object/bbox/xmin'].numpy().astype(float)[i]*image_width)
        xmax = int(example['image/object/bbox/xmax'].numpy().astype(float)[i]*image_width)
        ymin = int(example['image/object/bbox/ymin'].numpy().astype(float)[i]*image_height)
        ymax = int(example['image/object/bbox/ymax'].numpy().astype(float)[i]*image_height)
        conf = example['image/object/bbox/conf'].numpy().astype(float)[i]

        print(f"{i:d} {label:} ({xmin:.2f}, {ymin:.2f}) ({xmax:.2f}, {ymax:.2f})  (C={conf:.3f})")

        color = colors[label]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.putText(image, label[:1], (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 1)
    print("--")

    cv2.putText(image, "{:05d}".format(frame), 
                (10, image_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,255), 2)

    if args.scale != 1.0:
        image = cv2.resize(image, None, fx = 1 / args.scale, fy = 1 / args.scale)

    cv2.imshow("image", image)

    key = cv2.waitKey(0)

    if key in [ord("q")]:
        break

    cv2.waitKey(10)

    ct += 1
    if args.limit and ct >= limit:
        print('\n')
        print("Limit reached.  {} frames shown.".format(limit))
        print('\n')
        break

if not args.limit:
    print('\n')
    print("End of file.  {} frames shown.".format(ct))
    print('\n')

cv2.destroyAllWindows()
cv2.waitKey(10)

#!/usr/bin/env python 

import cv2
import tensorflow as tf
import numpy as np
import argparse

import train

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, help='Path of the tfrecord file.')
parser.add_argument('--limit', help='Number of frames to show', type = int)

args = parser.parse_args()
file_path = args.file

if args.limit:
    limit = args.limit


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

tfrecord_dataset = tf.data.TFRecordDataset([file_path])
parsed_dataset = tfrecord_dataset.map(train.read_tfrecord)
cv2.namedWindow("image")
cv2.startWindowThread()
ct = 0
for example in parsed_dataset:
    image_raw = example['image/encoded'].numpy()
    image_encoded = np.frombuffer(image_raw, np.uint8)
    image = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)
    image_width = example['image/width'].numpy()
    image_height = example['image/height'].numpy()
    labels = example['image/object/class/text'].numpy().astype(str)
    for i, label in enumerate(labels):
        xmin = int(example['image/object/bbox/xmin'].numpy().astype(float)[i]*image_width)
        xmax = int(example['image/object/bbox/xmax'].numpy().astype(float)[i]*image_width)
        ymin = int(example['image/object/bbox/ymin'].numpy().astype(float)[i]*image_height)
        ymax = int(example['image/object/bbox/ymax'].numpy().astype(float)[i]*image_height)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
        cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255,255,0), 2)

    frame = example['image/source_id'].numpy().decode("utf-8")
    print(frame, image.shape, image_width, image_height)
    cv2.putText(image, frame, (10, image_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
               1, (255,0,255), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
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

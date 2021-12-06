#!/usr/bin/env python

import cv2 
import os, sys, re
from time import sleep
from copy import deepcopy as dc

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from glob import glob

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)


label_dict = {"person" : 1, "bicycle" : 2, "car" : 3, "bus" : 6, "truck" : 8}
tag_dict = {"p" : "person", "i" : "bicycle", "c" : "car", "b" : "bus", "t" : "truck"}
cat_dict = {v : k for k, v in label_dict.items()}

colors = {"person" : (0, 0, 255), "bicycle" : (255, 0, 255), 
          "car"    : (255, 0, 0),
          "truck"  : (0, 255, 255),
          "bus"    : (0, 255, 0)}

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', "--ifile",  type = str, default = "intersections/210930_112919_train.tfrecord")
parser.add_argument('-o', "--ofile",  type = str, default = "")
parser.add_argument("-c", "--categs", default = ["car", "truck", "bus"], choices = label_dict, 
                    nargs = "+", help = "Types of objects to process -- must match labels")
parser.add_argument("-t", "--threshold", type = float, default = 0.1)
parser.add_argument("-e", "--write_empty", action = "store_true")
parser.add_argument("-s", "--skip_empty", action = "store_true")
parser.add_argument("-v", "--verbose", action = "store_true")
parser.add_argument("-w", "--overwrite", action = "store_true")
parser.add_argument("--display_height", type = int, default = 1900)
parser.add_argument("--display_width", type = int, default = 3600)

parser.add_argument("-k", "--keep_all", action = "store_true", help = "write all records, including un-sanitized")

args = parser.parse_args()
IFILE = args.ifile
CATEGS = args.categs
SKIP_EMPTY = args.skip_empty
VERBOSE= args.verbose
THRESH = args.threshold
NMS_THRESH = 0.3
OVERWRITE = args.overwrite
KEEP_ALL = args.keep_all
DISPLAY_HEIGHT = args.display_height
DISPLAY_WIDTH = args.display_width

WRITE_EMPTY = args.write_empty

if not args.ofile:
    args.ofile = IFILE.replace(".tfrecord", ".clean.tfrecord")

if os.path.exists(args.ofile) and not OVERWRITE:
    print(args.ofile, "exists -- exiting")
    sys.exit()

banner = f"""

#########################################
####  File: {IFILE}
#########################################
"""
print(banner)


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


def _parse_image_function(protob):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(protob, image_feature_description)


def get_detections_dict(detections):

    width  = detections['image/width'].numpy()
    height = detections['image/height'].numpy()

    sanitized = bool(detections['image/sanitized'].numpy())

    return {
      "width"     : width,
      "height"    : height,
      "sanitized" : sanitized,
      "label"     : detections['image/object/class/text'].numpy().astype(str),
      "xmin"      : detections['image/object/bbox/xmin'].numpy().astype(float),
      "xmax"      : detections['image/object/bbox/xmax'].numpy().astype(float),
      "ymin"      : detections['image/object/bbox/ymin'].numpy().astype(float),
      "ymax"      : detections['image/object/bbox/ymax'].numpy().astype(float),
      "conf"      : detections['image/object/bbox/conf'].numpy().astype(float)
    }


def draw_detections_on_image(image, detections, thresh = THRESH):

    d = detections

    height, width = image.shape[:2]

    for xmin, xmax, ymin, ymax, conf, label in \
        zip(d["xmin"], d["xmax"], d["ymin"], d["ymax"], d["conf"], d["label"]):

        if conf < thresh: continue
        if label not in CATEGS: continue

        color = colors[label]

        cv2.rectangle(image,
                      (int(xmin * width), int(ymin * height)), 
                      (int(xmax * width), int(ymax * height)), color, 1)

        cv2.putText(image, label[:1], (int(xmin * width), int(ymax * height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    return image

def print_detections(detections, verbose = True):

    if not verbose: return

    print("=====")

    d = detections
    print(" -->>", len(d["conf"]))

    for xmin, xmax, ymin, ymax, conf, label in \
        zip(d["xmin"], d["xmax"], d["ymin"], d["ymax"], d["conf"], d["label"]):

        print(f"{label:} ({xmin:.2f}, {ymin:.2f}) ({xmax:.2f}, {ymax:.2f})  (C={conf:.3f})")

    print("=====")


def _int64_feature(value):      return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):      return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def _int64_list_feature(value): return tf.train.Feature(int64_list = tf.train.Int64List(value = list(value)))
def _bytes_list_feature(value): return tf.train.Feature(bytes_list = tf.train.BytesList(value = list(value)))
def _float_list_feature(value): return tf.train.Feature(float_list = tf.train.FloatList(value = list(value)))


def filter_detections(dets, thresh):

    od = dc(dets)

    mask  = od["conf"] > thresh
    mask &= np.isin(od["label"], CATEGS)
    mask &= od["xmin"].round(2) < od["xmax"].round(2) 
    mask &= od["ymin"].round(2) < od["ymax"].round(2) 

    for v in ["xmin", "xmax", "ymin", "ymax", "conf", "label"]:
        od[v] = od[v][mask]

    return od


def make_record(original, detection_dict, thresh = THRESH):
    
    d = filter_detections(detection_dict, thresh)

    int_labels = [label_dict[txt] for txt in d["label"]]
    txt_labels = [txt.encode('utf-8') for txt in d["label"]]

    return tf.train.Example(features = tf.train.Features(feature={
        'image/video'              : _bytes_feature(original['image/video'].numpy()),
        'image/frame_id'           : _int64_feature(original['image/frame_id'].numpy()),
        'image/tag'                : _bytes_feature(original['image/tag'].numpy()),
        'image/timestamp'          : _bytes_feature(original['image/timestamp'].numpy()),
        'image/encoded'            : _bytes_feature(original["image/encoded"].numpy()),
        'image/format'             : _bytes_feature(original["image/format"].numpy()),

        'image/sanitized'          : _int64_feature(d["sanitized"]),

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


def append_detections(image, detections, scale = 1.0):

    roi_xmin, roi_ymin, roi_width, roi_height = cv2.selectROI("image", image)

    if not roi_width or not roi_height: return detections

    roi_xmax, roi_ymax = roi_xmin + roi_width, roi_ymin + roi_height

    img_height, img_width = image.shape[:2]

    roi_xmin /= img_width
    roi_xmax /= img_width
    roi_ymin /= img_height
    roi_ymax /= img_height


    if len(CATEGS) == 1: label = CATEGS[0]
    else:
        print("Is it a...", tag_dict)

        while True:
            key = cv2.waitKey(10)
            if key < 0: continue
            if chr(key) in tag_dict:
                label = tag_dict[chr(key)]
                break

    detections["xmin"]  = np.append(detections["xmin"],  roi_xmin)
    detections["xmax"]  = np.append(detections["xmax"],  roi_xmax)
    detections["ymin"]  = np.append(detections["ymin"],  roi_ymin)
    detections["ymax"]  = np.append(detections["ymax"],  roi_ymax)
    detections["conf"]  = np.append(detections["conf"],  1.)
    detections["label"] = np.append(detections["label"], label)

    return detections


def remove_nms_duplicates(image, detections):

    height, width = image.shape[:2]

    d = detections 
    confs = dc(d["conf"])
    boxes = [[int(xmin * width), int(ymin * height), 
              int((xmax - xmin) * width), int((ymax - ymin) * height)] 
             for xmin, xmax, ymin, ymax in \
             zip(d["xmin"], d["xmax"], d["ymin"], d["ymax"])]

    idxs = cv2.dnn.NMSBoxes(boxes, confs, THRESH, NMS_THRESH)

    od = {"width" : d["width"], "height" : d["height"], "sanitized" : d["sanitized"],
          "xmin" : [], "xmax" : [], "ymin" : [], "ymax" : [], "conf" : [], "label" : []}

    fields = ["xmin", "xmax", "ymin", "ymax", "conf", "label"]

    if len(idxs):

        for idx in idxs:
            for f in fields:
                od[f].append(d[f][idx])

    for f in fields:

        od[f] = np.array(od[f]).flatten()
        if f[0] in "xy":
            od[f] = od[f].astype(float)

    return od


def remove_detections(image, detections):

    roi_xmin, roi_ymin, roi_width, roi_height = cv2.selectROI("image", image)

    if not roi_width or not roi_height: return detections

    roi_xmax, roi_ymax = roi_xmin + roi_width, roi_ymin + roi_height

    img_height, img_width = image.shape[:2]

    roi_xmin /= img_width
    roi_xmax /= img_width
    roi_ymin /= img_height
    roi_ymax /= img_height

    d = detections 
    od = {"width" : d["width"], "height" : d["height"], "sanitized" : d["sanitized"],
          "xmin" : [], "xmax" : [], "ymin" : [], "ymax" : [], "conf" : [], "label" : []}

    for xmin, xmax, ymin, ymax, conf, label in \
        zip(d["xmin"], d["xmax"], d["ymin"], d["ymax"], d["conf"], d["label"]):

        x_intx = (xmax > roi_xmin) and (roi_xmax > xmin)
        y_intx = (ymax > roi_ymin) and (roi_ymax > ymin)

        if x_intx and y_intx: continue

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


def unique_in_roi(image, detections):

    roi_xmin, roi_ymin, roi_width, roi_height = cv2.selectROI("image", image)

    if not roi_width or not roi_height: return detections

    roi_xmax, roi_ymax = roi_xmin + roi_width, roi_ymin + roi_height

    img_height, img_width = image.shape[:2]

    roi_xmin /= img_width
    roi_xmax /= img_width
    roi_ymin /= img_height
    roi_ymax /= img_height

    d = detections 
    od = {"width" : d["width"], "height" : d["height"], "sanitized" : d["sanitized"],
          "xmin" : [], "xmax" : [], "ymin" : [], "ymax" : [], "conf" : [], "label" : []}

    roi_max_conf, roi_max_detection = 0, {}

    for xmin, xmax, ymin, ymax, conf, label in \
        zip(d["xmin"], d["xmax"], d["ymin"], d["ymax"], d["conf"], d["label"]):

        x_intx = (xmax > roi_xmin) and (roi_xmax > xmin)
        y_intx = (ymax > roi_ymin) and (roi_ymax > ymin)

        if x_intx and y_intx:
            if conf > roi_max_conf:
                roi_max_conf = conf
                roi_max_detection = {"xmin" : xmin, "xmax" : xmax,
                                     "ymin" : ymin, "ymax" : ymax, 
                                     "label" : label, "conf" : conf}
            continue

        od["xmin"].append(xmin)
        od["xmax"].append(xmax)
        od["ymin"].append(ymin)
        od["ymax"].append(ymax)
        od["conf"].append(conf)
        od["label"].append(label)

    if roi_max_conf > 0:
        for k, v in roi_max_detection.items():
            od[k].append(v)

    for v in ["xmin", "xmax", "ymin", "ymax", "conf", "label"]:

        od[v] = np.array(od[v])
        if v[0] in "xy":
            od[v] = od[v].astype(float)

    return od

def relabel_detection(image, detections):

    roi_xmin, roi_ymin, roi_width, roi_height = cv2.selectROI("image", image)

    if not roi_width or not roi_height: return detections

    roi_xmax, roi_ymax = roi_xmin + roi_width, roi_ymin + roi_height

    img_height, img_width = image.shape[:2]

    roi_xmin /= img_width
    roi_xmax /= img_width
    roi_ymin /= img_height
    roi_ymax /= img_height

    d = detections 
    od = {"width" : d["width"], "height" : d["height"], "sanitized" : d["sanitized"],
          "xmin" : [], "xmax" : [], "ymin" : [], "ymax" : [], "conf" : [], "label" : []}

    print("Is it a...", tag_dict)
    while True:
        key = cv2.waitKey(10)
        if key < 0: continue
        if chr(key) in tag_dict:
            new_label = tag_dict[chr(key)]
            break

    roi_max_conf, roi_max_detection = 0, {}

    for xmin, xmax, ymin, ymax, conf, label in \
        zip(d["xmin"], d["xmax"], d["ymin"], d["ymax"], d["conf"], d["label"]):

        x_intx = (xmax > roi_xmin) and (roi_xmax > xmin)
        y_intx = (ymax > roi_ymin) and (roi_ymax > ymin)

        if x_intx and y_intx: label = new_label

        od["xmin"].append(xmin)
        od["xmax"].append(xmax)
        od["ymin"].append(ymin)
        od["ymax"].append(ymax)
        od["conf"].append(conf)
        od["label"].append(label)

    if roi_max_conf > 0:
        for k, v in roi_max_detections:
            od[k].append(v)

    for v in ["xmin", "xmax", "ymin", "ymax", "conf", "label"]:

        od[v] = np.array(od[v])
        if v[0] in "xy":
            od[v] = od[v].astype(float)

    return od



def set_confidences_to_1(detections):

    detections["conf"] = np.ones(detections["conf"].size)

    return detections


tfrecord_dataset = tf.data.TFRecordDataset([IFILE])
parsed_dataset = tfrecord_dataset.map(_parse_image_function)
nrecords = len(list(parsed_dataset))

output_records = tf.io.TFRecordWriter(args.ofile)

cv2.namedWindow("image")
cv2.startWindowThread()
cv2.moveWindow("image", 950, 0)

QUIT = False
OK = False

n_output = 0
n_sanitized = 0
for frame in tqdm(parsed_dataset, total = nrecords):

    image_raw = frame['image/encoded'].numpy()
    image_encoded = np.frombuffer(image_raw, np.uint8)
    original_image = cv2.imdecode(image_encoded, cv2.IMREAD_COLOR)

    detections = get_detections_dict(frame)
    detections = remove_nms_duplicates(original_image, detections)

    if SKIP_EMPTY and not detections["conf"].size:
        continue

    if not QUIT: 
        print("sanitized:", detections["sanitized"])
        print_detections(detections, VERBOSE)


    OK = False
    HIDE = False

    while not QUIT and not detections["sanitized"]:

        image = original_image.copy()

        if not HIDE:
            image = draw_detections_on_image(image, detections, THRESH)

        SCALE  = min(DISPLAY_HEIGHT / detections["height"],
                     DISPLAY_WIDTH / detections["width"])

        width  = int(detections["width"]  * SCALE)
        height = int(detections["height"] * SCALE)

        frame_id = frame['image/frame_id'].numpy()
        cv2.putText(image, "{:05d}".format(frame_id), (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

        image = cv2.resize(image, None, fx = SCALE, fy = SCALE, interpolation = cv2.INTER_AREA)

        cv2.imshow("image", image)

        key = cv2.waitKey(10)

        # Play and pause to count entrances and exits
        if key == ord("q"):
            QUIT = True
            break

        if key == ord("s"):
            OK = False
            break

        if key == ord("o"):
            OK = True

        elif key == ord("k") and OK:
            
            # detections = set_confidences_to_1(detections)
            detections["sanitized"] = True
            n_sanitized += 1
            print(n_sanitized, "sanitized")
            print_detections(detections, VERBOSE)

            break

        elif key == ord('h'):

            HIDE = False if HIDE else True

        elif key == ord("="):

            detections = append_detections(image, detections, SCALE)
            print_detections(detections, VERBOSE)

        elif key == ord("-"):

            detections = remove_detections(image, detections)
            print_detections(detections, VERBOSE)

        elif key == ord("1"):

            detections = unique_in_roi(image, detections)
            print_detections(detections, VERBOSE)

        elif key == ord("l"):

            detections = relabel_detection(image, detections)
            print_detections(detections, VERBOSE)

        elif key == ord("p"):

            print_detections(detections)

        elif key == 46:

            THRESH = round(THRESH + 0.01, 2)
            print("Threshold is now:", THRESH)

        elif key == 44:

            THRESH = round(THRESH - 0.01, 2)
            print("Threshold is now:", THRESH)

        elif key >= 0: OK = False

    if not QUIT and not OK and not detections["sanitized"]: 
        print("Frame skipped.")
        continue

    if QUIT and not KEEP_ALL: break

    if not WRITE_EMPTY and \
       not filter_detections(detections, THRESH)["conf"].size:

        print("no records to write.  getting out.")
        continue

    if not QUIT:
        print("Detections in frame:", (detections["conf"] > THRESH).sum())

    record = make_record(frame, detections)

    output_records.write(record.SerializeToString())

    n_output += 1

    if not QUIT: print("Records written:", n_output)


output_records.close()

cv2.destroyAllWindows()
cv2.waitKey(10)

sys.exit()




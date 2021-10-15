#!/usr/bin/env python

import os, sys

import tqdm

import tensorflow as tf
print(tf.__version__)
tf.get_logger().setLevel('WARNING')

import numpy as np
import matplotlib

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

import pandas as pd

import cv2
from PIL import Image


cv2pil = lambda x : Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
np2pil = lambda x : Image.fromarray(x)


## Load the labels.
def get_label_map(path = 'models/research/object_detection/data/mscoco_label_map.pbtxt'):

    label_map = label_map_util.load_labelmap(path)
    label_map = label_map_util.get_label_map_dict(label_map, use_display_name = True)

    return label_map

## Get the model configuration and checkpoint.
def get_model_detection_function(model = "centernet_hourglass104_512x512_coco17_tpu-8"):
    """Get a tf.function for detection."""

    pipeline_config = f'models/research/object_detection/configs/tf2/{model}.config'
    model_dir       = f'models/research/object_detection/test_data/checkpoint/{model}/'
    
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    
    model_config    = configs['model']
    detection_model = model_builder.build(model_config = model_config, is_training = False)
    
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()
  
    @tf.function(experimental_relax_shapes=True)
    def detect_fn(image):
        """Detect objects in image."""
    
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
    
        return detections, prediction_dict, tf.reshape(shapes, [-1])
  
    return detect_fn

CATEGORY_OFFSET = 1

def detections_to_list(img, detections, thresh = 0.3, 
                       categories = ["person"], 
                       video_name = "X", frame_id = 0,
                       roi = None):
    '''
    Converts a dictionary of features for a single frame to a tf_example object.
    '''
    
    ## Fixed values to pass in.
    boxes   = detections['detection_boxes']  [0]
    scores  = detections['detection_scores'] [0]
    classes = detections['detection_classes'][0]

    classes = np.array(classes).astype(int) + CATEGORY_OFFSET

    keep_categories = {label_map[k] : k for k in categories}
    keep_detections = np.isin(classes, list(keep_categories)) & (scores > thresh)

    if not keep_detections.numpy().any(): return []
    
    boxes = boxes[keep_detections]
    classes = list(classes[keep_detections])
    scores = scores[keep_detections]
    scores = list(np.array(scores).astype(float))

    classes_text = []
    for label in classes:
        classes_text.append(keep_categories[label])
                                   
    np_box = np.array(boxes).astype(float)

    xmin, ymin = 0, 0
    rangex, rangey = 1, 1

    if roi:
        xmin, xmax, ymin, ymax = roi
        rangex = xmax - xmin
        rangey = ymax - ymin

    records = []
    for label, conf, bbox in zip(classes_text, scores, np_box):

        det = {"frame" : frame_id,
               "label" : label, "conf" : conf,
               "xmin" : xmin + rangex * bbox[1], 
               "xmax" : xmin + rangex * bbox[3], 
               "ymin" : ymin + rangey * bbox[0],
               "ymax" : ymin + rangey * bbox[2]}

        records.append(det)
                               
    return records


def records_from_stream(video, ouput, THRESH = 0.35, ROI = [], CATEGS = ["person"], N = 100):

    cap = cv2.VideoCapture(video)

    XOFFSET, YOFFSET = 0, 0

    if ROI:

        WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        XMIN_CLIP, XMAX_CLIP = int(WIDTH  * ROI[0]), int(WIDTH  * ROI[1])
        YMIN_CLIP, YMAX_CLIP = int(HEIGHT * ROI[2]), int(HEIGHT * ROI[3])


    all_detections = []

    show_tqdm_bar = 0 if N > 999999 else N
    for ix in tqdm.tqdm(range(N), total = show_tqdm_bar):

        ret, cv_img = cap.read()

        if not ret: break

        if ROI:
            cv_img = cv_img[YMIN_CLIP:YMAX_CLIP,XMIN_CLIP:XMAX_CLIP,:]

        pil_img = cv2pil(cv_img)
        np_img  = np.array(pil_img)
        tf_img  = tf.convert_to_tensor(np.expand_dims(np_img, 0), dtype=tf.float32)

        detections, predictions_dict, shapes = detect_fn(tf_img)

        records = detections_to_list(cv_img, detections, thresh = THRESH,
                                     video_name = video, frame_id = ix,
                                     categories = CATEGS, roi = ROI)

        all_detections.extend(records)

    cap.release()

    df = pd.DataFrame(all_detections)
    df.to_csv(ouput + "_det.csv", index = False, float_format = "%.3f")

    

global detect_fn, label_map
def tag_videos(video, model, roi, thresh, total, categs):

    global detect_fn, label_map
    detect_fn = get_model_detection_function(model)
    label_map = get_label_map()

    output = video.replace(".mp4", ".csv")

    records_from_stream(video, output, THRESH = thresh, ROI = roi, N = total, CATEGS = categs)



if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  required = True, type = str)
    parser.add_argument('--total',  default = 1000000, help = 'How many frames', type = int)
    parser.add_argument("--roi",    default = [], type = float, nargs = 4, help = "xmin, xmax, ymin, ymax")
    parser.add_argument("--thresh", default = 0.25, type = float, help = "confidence threshold")
    parser.add_argument("--categs", default = ["car"], nargs = "+", help = "Types of objects to process -- must match labels")
    parser.add_argument('--model',  type = str, default = "centernet_hourglass104_512x512_coco17_tpu-8")
    args = parser.parse_args()

    tag_videos(**vars(args))



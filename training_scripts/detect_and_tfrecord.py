#!/usr/bin/env python

import os, sys
sys.path.append("/media/jsaxon/brobdingnag/projects/urban_vision/models")
sys.path.append("/media/jsaxon/brobdingnag/projects/urban_vision/models/research")

import tqdm

from glob import glob

import tensorflow as tf
print(tf.__version__)
tf.get_logger().setLevel('WARNING')

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import numpy as np
import matplotlib
import random

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

import cv2
from PIL import Image


cv2pil = lambda x : Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
np2pil = lambda x : Image.fromarray(x)

roi_dict = {

  '1l/90_33st_NB1'        : [0.01, 1.00, 0.08, 1.00],
  '1l/90_33st_NB2'        : [0.00, 0.85, 0.01, 1.00],
  '1l/90_33st_NB4'        : [0.00, 0.90, 0.00, 1.00],
  '1l/90_33st_NB5'        : [0.00, 1.00, 0.01, 1.00],
  '1l/90_33st_NB6'        : [0.00, 0.88, 0.01, 1.00],

  '1w/clark_grand'        : [0.34, 0.88, 0.09, 0.56],
  '1w/lower_wacker'       : [0.00, 0.57, 0.14, 1.00],
  '1w/state_adams_e'      : [0.00, 0.78, 0.09, 1.00],
  '1w/state_jackson'      : [0.00, 0.83, 0.12, 1.00],
  '1w/state_monroe_w'     : [0.26, 1.00, 0.15, 1.00],

  'intx/clark_division'   : [0.18, 0.99, 0.05, 1.00],
  'intx/ohio_orleans'     : [0.12, 1.00, 0.06, 1.00],
  'intx/sheridan_wilson'  : [0.04, 1.00, 0.11, 1.00],
  'intx/state_jackson'    : [0.02, 1.00, 0.06, 1.00],
  'intx/wabash_roosevelt' : [0.00, 1.00, 0.08, 1.00],

  'lf/106'                : [0.01, 0.59, 0.27, 0.96],
  'lf/108'                : [0.18, 0.85, 0.12, 0.97],
  'lf/109'                : [0.30, 0.96, 0.16, 1.00],
  'lf/120'                : [0.23, 0.97, 0.16, 0.96],
  'lf/213'                : [0.07, 0.72, 0.25, 0.94]

}


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

def _int64_feature(value):      return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):      return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def _int64_list_feature(value): return tf.train.Feature(int64_list = tf.train.Int64List(value =  value ))
def _bytes_list_feature(value): return tf.train.Feature(bytes_list = tf.train.BytesList(value =  value ))
def _float_list_feature(value): return tf.train.Feature(float_list = tf.train.FloatList(value =  value ))


def create_tfrecord(img, detections, thresh = 0.3, 
                    categories = ["person"], 
                    video_name = "X", frame_id = 0):
    '''
    Converts a dictionary of features for a single frame to a tf_example object.
    '''
    
    ## Fixed values to pass in.
    video         = str.encode(video_name)
    source_id     = str.encode(str("{:05d}".format(frame_id)))
    
    height        = img.shape[0]
    width         = img.shape[1]

    image_format  = str.encode('jpg')
    
    _, image_buff = cv2.imencode(".jpg", img.copy())
    image_encoded = image_buff.tobytes()
    
    boxes   = detections['detection_boxes']  [0]
    scores  = detections['detection_scores'] [0]
    classes = detections['detection_classes'][0]

    classes = np.array(classes).astype(int) + CATEGORY_OFFSET

    keep_categories = {label_map[k] : k for k in categories}
    keep_detections = np.isin(classes, list(keep_categories)) & (scores > thresh)

    if not keep_detections.numpy().any(): return False
    
    boxes = boxes[keep_detections]
    classes = list(classes[keep_detections])

    classes_text = []
    for label in classes:
        classes_text.append(keep_categories[label].encode('utf-8'))
                                   
    np_box = np.array(boxes).astype(float)
    xmins = list(np_box[:,1]) 
    xmaxs = list(np_box[:,3])
    ymins = list(np_box[:,0]) 
    ymaxs = list(np_box[:,2]) 

    scores = list(np.array(scores[keep_detections]).astype(float))
                               
    record = tf.train.Example(features = tf.train.Features(feature={
        'image/video'              : _bytes_feature(video),
        'image/height'             : _int64_feature(height),
        'image/width'              : _int64_feature(width),
        'image/source_id'          : _bytes_feature(source_id),
        'image/encoded'            : _bytes_feature(image_encoded),
        'image/format'             : _bytes_feature(image_format),
        'image/object/bbox/xmin'   : _float_list_feature(xmins),
        'image/object/bbox/xmax'   : _float_list_feature(xmaxs),
        'image/object/bbox/ymin'   : _float_list_feature(ymins),
        'image/object/bbox/ymax'   : _float_list_feature(ymaxs),
        'image/object/bbox/conf'   : _float_list_feature(scores),
        'image/object/class/label' : _int64_list_feature(classes),
        'image/object/class/text'  : _bytes_list_feature(classes_text),
    }))
                               
    return record


def paint_detections(img, detections, categories = ["person"], thresh = 0.3, category_offset = 1):
    
    img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape

    scores  = detections['detection_scores'] [0]
    classes = detections['detection_classes'][0]
    boxes   = detections['detection_boxes']  [0]

    keep_categories = [label_map[k] - category_offset for k in categories]

    display = np.isin(classes, keep_categories) & (scores > thresh)
    
    boxes   = boxes  [display]
    scores  = scores [display]
    classes = classes[display]

    COLOR_CAT = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
    for box, cat in zip(boxes, classes):
        
        ymin, xmin, ymax, xmax = np.array(box)
        
        cv2.rectangle(img, tuple((int(xmin * width), int(ymin * height))),
                           tuple((int(xmax * width), int(ymax * height))),
                      COLOR_CAT[cat], 2)

    return img



def records_from_stream(video, ouput, THRESH = 0.35, ROI = [], JITTER = 0, CATEGS = ["person"], N = 100, NSKIP = 10, VAL = 5, show = False):

    cap = cv2.VideoCapture(video)

    if ROI:

        WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        XMIN = int(WIDTH  * ROI[0])
        XMAX = int(WIDTH  * ROI[1])
        YMIN = int(HEIGHT * ROI[2])
        YMAX = int(HEIGHT * ROI[3])

    else:

        XMIN, YMIN = 0, 0

    jXMIN, jXMAX, jYMIN, jYMAX = 0, 0, 0, 0

    cap.read()

    if VAL:
        training   = tf.io.TFRecordWriter(ouput + "_train.tfrecord")
        validation = tf.io.TFRecordWriter(ouput + "_val.tfrecord")
    else:
        all_output = tf.io.TFRecordWriter(ouput + ".tfrecord")

    show_tqdm_bar = 0 if N > 999999 else N
    for ix in tqdm.tqdm(range(N), total = show_tqdm_bar):

        for xi in range(NSKIP): 
            ret, cv_img = cap.read()

            if not ret: break
        if not ret: break

        if ROI: 

            if JITTER:
                jXMIN = int(WIDTH  * JITTER * (np.random.rand() * 2 - 1))
                jXMAX = int(WIDTH  * JITTER * (np.random.rand() * 2 - 1))
                jYMIN = int(HEIGHT * JITTER * (np.random.rand() * 2 - 1))
                jYMAX = int(HEIGHT * JITTER * (np.random.rand() * 2 - 1))

                if jXMIN + XMIN < 0:      jXMIN = -XMIN
                if jXMAX + XMAX > WIDTH:  jXMIN = WIDTH  - XMAX - 1
                if jYMIN + YMIN < 0:      jYMIN = -YMIN
                if jYMAX + YMAX > HEIGHT: jYMIN = HEIGHT - YMAX - 1


            cv_img = cv_img[YMIN+jYMIN:YMAX+jYMAX,XMIN+jXMIN:XMAX+jXMAX,:]

        pil_img = cv2pil(cv_img)
        np_img  = np.array(pil_img)
        tf_img  = tf.convert_to_tensor(np.expand_dims(np_img, 0), dtype=tf.float32)

        detections, predictions_dict, shapes = detect_fn(tf_img)

        record = create_tfrecord(cv_img, detections, thresh = THRESH, 
                                 categories = CATEGS,
                                 video_name = video, 
                                 frame_id = (ix+1) * NSKIP)

        if not record: continue
        
        if VAL:
            if ix % VAL: training  .write(record.SerializeToString())
            else:        validation.write(record.SerializeToString())
        else:
            all_output.write(record.SerializeToString())


        if show: 
            img_det = paint_detections(np_img, detections, thresh = THRESH)
            img_det = cv2.resize(img_det, None, fx = 1 / 2, fy = 1/2)

            cv2.imshow('detections', img_det)
            if (cv2.waitKey(30) & 0xff) == 27: break

        
    if VAL:
        training  .close()
        validation.close()
    else:
        all_output.close()
    
    cap.release()
    

global detect_fn, label_map
def tag_videos(videos, model, jitter, thresh, total, skip, show, val, categs):

    global detect_fn, label_map
    detect_fn = get_model_detection_function(model)
    label_map = get_label_map()

    video_files = glob(videos)
    video_files = random.choices(video_files, k = 10)
    print(video_files)

    for video in video_files:

        output = video.replace(".mp4", "").replace("vids", "tfrecords")

        ofile = "/".join(output.split("/")[:-1])

        os.makedirs(ofile, exist_ok = True)

        print(video, output, ofile)

        for tag in roi_dict:
            if tag in video:
                roi = roi_dict[tag]

        records_from_stream(video, output, THRESH = thresh, ROI = roi, JITTER = jitter, 
                            N = total, NSKIP = skip,
                            VAL = val, CATEGS = categs, show = show)


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', required = True, type = str)
    parser.add_argument('--model',  type = str, default = "centernet_hourglass104_512x512_coco17_tpu-8")
    parser.add_argument('--total',  default = 1000000, help = 'How many frames', type = int)
    parser.add_argument('--skip',   default = 150, help = "1 in how many?", type = int)
    parser.add_argument('--jitter', default = 0.0, help = "how much to move around the roi?", type = float)
    parser.add_argument('--show',   action = "store_true", default = False)
    parser.add_argument("--thresh", default = 0.3, type = float, help = "confidence threshold")
    parser.add_argument("--val",   type = int, default = 0, help = "1 out of how many, to write to val?")
    parser.add_argument("--categs", default = ["car", "truck", "bus"], nargs = "+", help = "Types of objects to process -- must match labels")
    args = parser.parse_args()

    tag_videos(**vars(args))



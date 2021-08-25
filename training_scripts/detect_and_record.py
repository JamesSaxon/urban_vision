#!/usr/bin/env python

import os, sys
sys.path.append("/media/jsaxon/brobdingnag/projects/urban_vision/models")
sys.path.append("/media/jsaxon/brobdingnag/projects/urban_vision/models/research")

import tqdm

import tensorflow as tf
print(tf.__version__)
tf.get_logger().setLevel('WARNING')

import numpy as np
import matplotlib

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

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

    model = "centernet_hourglass104_512x512_coco17_tpu-8"
    
    pipeline_config = f'models/research/object_detection/configs/tf2/{model}.config'
    model_dir       = f'models/research/object_detection/test_data/checkpoint/'
    
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    
    model_config    = configs['model']
    detection_model = model_builder.build(model_config = model_config, is_training = False)
    
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()
  
    @tf.function
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
                    categories = ["person", "bicycle"], 
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
    
    scores  = detections['detection_scores'] [0]
    classes = detections['detection_classes'][0]
    boxes   = detections['detection_boxes']  [0]

    keep_categories = {label_map[k] - CATEGORY_OFFSET : k
                       for k in categories}
    
    keep_detections = np.isin(classes, list(keep_categories)) & (scores > thresh)
    
    boxes   = boxes  [keep_detections]
    scores  = scores [keep_detections]
    classes = classes[keep_detections]
                                   
    np_box = np.array(boxes).astype(float)
    xmins = list(np_box[:,1]) 
    xmaxs = list(np_box[:,3])
    ymins = list(np_box[:,0]) 
    ymaxs = list(np_box[:,2]) 

    classes_text = []
    for label in np.array(classes).astype(int):
        classes_text.append(keep_categories[label].encode('utf-8'))
    
    classes = list(np.array(classes).astype(int) + CATEGORY_OFFSET)
                               
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
        'image/object/class/label' : _int64_list_feature(classes),
        'image/object/class/text'  : _bytes_list_feature(classes_text),
    }))
                               
    return record


def paint_detections(img, detections, categories = ["person", "bicycle"], thresh = 0.3, category_offset = 1):
    
    thresh = 0.3

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



def tfrecords_from_stream(video, ouput, N = 100, NSKIP = 10, VAL = 5, show = False):

    cap = cv2.VideoCapture(video)
    ret, cv_img = cap.read()

    training   = tf.io.TFRecordWriter(ouput + "_train.tfrecord")
    validation = tf.io.TFRecordWriter(ouput + "_val.tfrecord")

    show_tqdm_bar = 0 if N > 999999 else N
    for ix in tqdm.tqdm(range(N), total = show_tqdm_bar):

        for xi in range(NSKIP): 
            ret, cv_img = cap.read()

            if not ret: break
        if not ret: break

        pil_img = cv2pil(cv_img)
        np_img  = np.array(pil_img)
        tf_img  = tf.convert_to_tensor(np.expand_dims(np_img, 0), dtype=tf.float32)

        detections, predictions_dict, shapes = detect_fn(tf_img)

        record = create_tfrecord(cv_img, detections, thresh = 0.5, 
                                 video_name = video, frame_id = (ix+1) * NSKIP)
        
        if ix % VAL: training  .write(record.SerializeToString())
        else:        validation.write(record.SerializeToString())

        if show: 
            img_det = paint_detections(np_img, detections, thresh = 0.5)
            img_det = cv2.resize(img_det, None, fx = 1 / 2, fy = 1/2)

            if (cv2.waitKey(30) & 0xff) == 27: break
            cv2.imshow('detections', img_det)

        
    training  .close()
    validation.close()
    
    cap.release()
    

global detect_fn, label_map
def tag_videos(video, output, total, skip, show):

    global detect_fn, label_map
    detect_fn = get_model_detection_function()
    label_map = get_label_map()

    tfrecords_from_stream(video, output, N = total, NSKIP = skip, show = show)



if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  required = True, type = str)
    parser.add_argument('--output', required = True, type = str)
    parser.add_argument('--total',  default = 1000000, help = 'How many frames', type = int)
    parser.add_argument('--skip',   default = 10, help = "1 in how many?", type = int)
    parser.add_argument('--show',   action = "store_true", default = False)
    args = parser.parse_args()

    tag_videos(**vars(args))



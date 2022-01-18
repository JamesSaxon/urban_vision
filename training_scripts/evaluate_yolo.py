#!/usr/bin/env python 

import cv2
import numpy as np
import argparse
import os

from tqdm import tqdm
from glob import glob

import configargparse


parser = configargparse.ArgParser(default_config_files = ['conf/stream_defaults.conf'])
parser.add("--yolo_size",  default = 608, type = int, choices = [320, 416, 608], help = "Detector size.")
parser.add("--yolo_path",  default = "../yolo/v3", type = str, help = "Path to YOLO directory.")
parser.add("--nms_thresh", default = 0.3, type = float, help = "NMS threshoold")
parser.add("--input",      required = True, type = str, help = "Directory from which to draw the images.")
parser.add("--output",     required = True, type = str, help = "Directory in which to place the detections.")
args = parser.parse_args()


odir = args.output
if not os.path.isdir(odir):
    print("creating directory:", odir)
    os.makedirs(odir, exist_ok = True)

yolo_path    = os.path.abspath(args.yolo_path)
yolo_config  = yolo_path + "/cfg"
yolo_weights = yolo_path + "/wgts"
yolo_labels  = yolo_path + "/names"
yolo_size    = args.yolo_size

yolo_model = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

yn = yolo_model.getLayerNames()
yolo_names = [yn[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]

yolo_labels = open(yolo_labels).read().strip().split("\n")

nms_thresh = args.nms_thresh

images = glob(args.input + "/*jpg")


for image_name in tqdm(images):

    frame = cv2.imread(image_name)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (yolo_size, yolo_size),
                                 swapRB = True, crop = False)
    
    yolo_model.setInput(blob)
    
    yolo_detections = yolo_model.forward(yolo_names)
    
    raw_boxes, nms_boxes, confs, classes = [], [], [], []
    
    # loop over each of the layer outputs
    for output in yolo_detections:
    
        # loop over each of the detections
        for det in output:
        
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = det[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            
            # Do not filter weak detections, here.
            if conf < 0.02: continue
            
            # YOLO returns center + dimensions.
            # scale the bbox to to image size
            raw_x, raw_y, raw_w, raw_h = det[0:4]

            box = det[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
            (int_x, int_y, nms_w, nms_h) = box.astype("int")
                
            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            nms_x = int(int_x - (nms_w / 2)) 
            nms_y = int(int_y - (nms_h / 2))
            nms_w = int(nms_w) ## python int, not np int!!! yeesh!
            nms_h = int(nms_h) ## python int, not np int!!! yeesh!
                
            # Update lists needed for NMS.
            raw_boxes.append([raw_x, raw_y, raw_w, raw_h])
            nms_boxes.append([nms_x, nms_y, nms_w, nms_h])
            confs    .append(float(conf))
            classes  .append(class_id)
    
    idxs = cv2.dnn.NMSBoxes(nms_boxes, confs, 0.6, 0.3)
    
    if not len(idxs): idxs = []
    else: idxs = set(idxs.flatten())
    

    txt_name = image_name.split("/")[-1].replace("jpg", "txt")
    with open(odir + txt_name, "w") as out:

        for i in idxs:
        
            class_id = classes[i]
            label = yolo_labels[class_id]

            conf = confs[i]
        
            x, y, w, h = raw_boxes[i]
        
            out.write(f"{label} {conf:.5f} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
    
    


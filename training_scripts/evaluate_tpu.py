#!/usr/bin/env python 

import cv2
import numpy as np
import argparse
import os

from tqdm import tqdm
from glob import glob

import configargparse

from PIL import Image

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils


parser = configargparse.ArgParser()
parser.add("--model",      required = True, type = str, help = "Detector.")
parser.add("--labels",     required = True, type = str, help = "Label file.")
parser.add("--nms_thresh", default = 0.3, type = float, help = "NMS threshoold")
parser.add("--input",      required = True, type = str, help = "Directory from which to draw the images.")
parser.add("--output",     required = True, type = str, help = "Directory in which to place the detections.")
args = parser.parse_args()


odir = args.output
if not os.path.isdir(odir):
    print("creating directory:", odir)
    os.makedirs(odir, exist_ok = True)

engine = DetectionEngine(args.model)
labels = dataset_utils.read_label_file(args.labels) 
nms_thresh = args.nms_thresh


images = glob(args.input + "/*jpg")

for image_name in tqdm(images):

    frame = cv2.imread(image_name)

    frame_height, frame_width = frame.shape[:2]

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    raw_detections = engine.detect_with_image(image, threshold = 0.01,
                                              keep_aspect_ratio = False, 
                                              relative_coord = False, top_k = 100)

    ctr_boxes, nms_boxes, confs, classes = [], [], [], []

    for iobj, obj in enumerate(raw_detections):

        # Get the extents.... 
        xmin, ymin, xmax, ymax = obj.bounding_box.flatten()

        # Convert formats...
        nms_x = xmin
        nms_y = ymin
        nms_w = xmax - xmin
        nms_h = ymax - ymin

        ctr_x = (xmin + xmax) / 2 / frame_width
        ctr_y = (ymin + ymax) / 2 / frame_height
        ctr_w = nms_w / frame_width
        ctr_h = nms_h / frame_height
            
        # Update lists needed for NMS.
        ctr_boxes.append([ctr_x, ctr_y, ctr_w, ctr_h])
        nms_boxes.append([nms_x, nms_y, nms_w, nms_h])
        confs    .append(float(obj.score))
        classes  .append(labels[obj.label_id])

    idxs = cv2.dnn.NMSBoxes(nms_boxes, confs, 0.6, 0.3)
    
    if not len(idxs): idxs = []
    else: idxs = set(idxs.flatten())

    txt_name = image_name.split("/")[-1].replace("jpg", "txt")
    with open(odir + txt_name, "w") as out:

        for i in idxs:
        
            label = classes[i]
            conf  = confs[i]
        
            x, y, w, h = ctr_boxes[i]
        
            out.write(f"{label} {conf:.5f} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
    


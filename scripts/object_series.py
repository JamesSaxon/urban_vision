#!/usr/bin/env python 

import argparse
import platform
import subprocess

import cv2

import numpy as np

from PIL import Image
from PIL import ImageDraw

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from glob import glob

import sys, tqdm

import pandas as pd

import tracker as tr, detector as det

colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = "../models/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite", help = "The tflite model")
    parser.add_argument('--labels', default = "../models/coco_labels.txt", help='Path of the labels file.')
    parser.add_argument('--input', help = 'Directory of input images.', required=True)
    parser.add_argument('-k', default = 10, type = int)
    parser.add_argument('--thresh', default = 0.5, type = float)
    parser.add_argument("--categs", default = [], nargs = "+")
    parser.add_argument("--frames", default = 0, type = int)
    parser.add_argument("--predict_matches", default = False, action = "store_true")
    parser.add_argument("--max_missing", default = 5, type = int)
    parser.add_argument("--max_distance", default = 0.125, type = float)
    parser.add_argument("--max_overlap", default = 0.25, type = float)
    parser.add_argument("--min_obs", default = 5, type = float)
    parser.add_argument("--contrail", default = 0, type = int)
    parser.add_argument("--select_roi", default = False, action = "store_true")
    parser.add_argument("--roi", default = [], type = float, nargs = 4)
    parser.add_argument("--roi_loc", default = "upper center", type = str)
    parser.add_argument("--edge_veto", default = 0, type = float)
    parser.add_argument("--view", default = False, action = "store_true")
    parser.add_argument("--scale", default = 1, type = float)
    parser.add_argument("--verbose", default = False, action = "store_true")
    parser.add_argument("--no_output", default = False, action = "store_true")
    parser.add_argument("--kalman", default = 50, type = int, help = "Scale of Kalman error covariance in pixels.")
  
    parser.add_argument('--keep_aspect_ratio', default = False, dest='keep_aspect_ratio', action='store_true',)
  
    args = parser.parse_args()
  
    # Initialize engine.
    vid = cv2.VideoCapture(args.input)

    tracker = tr.Tracker(max_missing = args.max_missing,
                         max_distance = args.max_distance,
                         predict_match_locations = args.predict_matches,
                         kalman_cov = args.kalman,
                         contrail = args.contrail)

    detector = det.Detector(model = args.model, labels = args.labels, 
                            categs = args.categs, thresh = args.thresh, k = args.k, 
                            max_overlap = args.max_overlap,
                            loc = args.roi_loc, edge_veto = args.edge_veto, 
                            verbose = False)

    vid = cv2.VideoCapture(args.input)

    if not args.no_output:
    
        out = cv2.VideoWriter(args.input.replace("gif", "mov").replace("mov", "mp4").replace(".mp4", "_det.mp4"), 
                              cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                              (round(vid.get(3) / args.scale), round(vid.get(4) / args.scale)))
    else: out = None


    ROI = []
    if args.select_roi or len(args.roi):

        ret, img = vid.read()
        vid.release()

        if not ret: 
            print("Video file not valid")
            sys.exit()

        print("frame size", img.shape[1], img.shape[0])

        if args.select_roi:

            if args.scale != 1:
                scaled = cv2.resize(img, None,
                                    fx = 1 / args.scale, fy = 1 / args.scale,
                                    interpolation = cv2.INTER_AREA)

            ROI = cv2.selectROI(scaled)
            cv2.destroyWindow("ROI selector")

            ROI = [int(x * args.scale) for x in ROI]

            XMIN, XMAX = ROI[0], ROI[0] + ROI[2]
            YMIN, YMAX = ROI[1], ROI[1] + ROI[3]

            print(XMIN / img.shape[1], 
                  XMAX / img.shape[1], 
                  YMIN / img.shape[0],
                  YMAX / img.shape[0])

        else: 

            XMIN = int(img.shape[1] * args.roi[0])
            XMAX = int(img.shape[1] * args.roi[1])
            YMIN = int(img.shape[0] * args.roi[2])
            YMAX = int(img.shape[0] * args.roi[3])

        ROI = [XMIN, XMAX, YMIN, YMAX]

        shade = 2 * np.ones((int(img.shape[0] / args.scale), int(img.shape[1] / args.scale))).astype("uint8")
        shade[int(YMIN/args.scale):int(YMAX/args.scale),int(XMIN/args.scale):int(XMAX/args.scale)] -= 1

        tracker.set_roi({"xmin" : XMIN, "xmax" : XMAX, "ymin" : YMIN, "ymax" : YMAX},
                        roi_buffer = (YMAX - YMIN) * 0.02)

        # Re-set...
        vid = cv2.VideoCapture(args.input)


    nframe = 0
    while True:
  
        ret, frame = vid.read()
  
        nframe += 1

        print(nframe, end = " ", flush = True)

        if not ret: break
        if args.frames and nframe > args.frames: break

        detections = detector.detect(frame, [XMIN, YMIN, XMAX, YMAX], return_image = True)

        scaled = cv2.resize(detections["image"],
                            None, fx = 1 / args.scale, fy = 1 / args.scale,
                            interpolation = cv2.INTER_AREA)

        if ROI: scaled = (scaled / shade[:,:,np.newaxis]).astype("uint8")

        tracker.update(detections["xy"], detections["boxes"], detections["areas"], detections["confs"])
        tracker.draw(scaled, scale = args.scale, min_obs = args.min_obs)

        if args.view:
        
            cv2.imshow("view", scaled)
            if (cv2.waitKey(1) & 0xff) == 27: break

        if out is not None:

            scaled = cv2.putText(scaled, "{:05d}".format(nframe), org = (120, 120), 
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2,
                                color = (255, 255, 255), thickness = 2)

            out.write(scaled)
    
        if args.verbose: print('Recorded frame {}'.format(nframe))
  

    if out is not None: out.release()
    tracker.write(args.input.split(".")[0] + "_tracker.csv")


if __name__ == '__main__': main()




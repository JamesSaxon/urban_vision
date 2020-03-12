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

from tracker import *

colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument('--input', help = 'Directory of input images.', required=True)
    parser.add_argument('-k', default = 10, type = int)
    parser.add_argument('--thresh', default = 0.5, type = float)
    parser.add_argument("--categs", default = [], nargs = "+")
    parser.add_argument("--frames", default = 0, type = int)
    parser.add_argument("--contrail", default = 0, type = int)
    parser.add_argument("--roi", default = False, action = "store_true")
    parser.add_argument("--view", default = False, action = "store_true")
    parser.add_argument("--scale", default = 1, type = float)
    parser.add_argument("--verbose", default = False, action = "store_true")
    parser.add_argument("--no_output", default = False, action = "store_true")
    parser.add_argument("--kalman", default = 50, type = int, help = "Scale of Kalman error covariance in pixels.")
  
    parser.add_argument('--keep_aspect_ratio', default = False, dest='keep_aspect_ratio', action='store_true',)
  
    args = parser.parse_args()
  
    # Initialize engine.
    engine = DetectionEngine(args.model)
    labels = dataset_utils.read_label_file(args.label) if args.label else None
  
    vid = cv2.VideoCapture(args.input)

    tracker = Tracker(max_missing = 5, max_distance = 250, contrail = args.contrail)

    if args.roi:

        ret, img = vid.read()
        vid.release()

        if not ret: 
            print("Video file not valid")
            sys.exit()

        if args.scale != 1:
            img = cv2.resize(img, None, fx = 1 / args.scale, fy = 1 / args.scale, interpolation = cv2.INTER_AREA)

        ROI = cv2.selectROI(img)
        cv2.destroyWindow("ROI selector")

        ROI = [int(x * args.scale) for x in ROI]

        XMIN, XMAX = ROI[0], ROI[0] + ROI[2]
        YMIN, YMAX = ROI[1], ROI[1] + ROI[3]

        vid = cv2.VideoCapture(args.input)


    out = None
    if not args.no_output:
    
        out = cv2.VideoWriter(args.input.replace("gif", "mov").replace("mov", "mp4").replace(".mp4", "_det.mp4"), 
                              cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                              (round(vid.get(3) / args.scale), round(vid.get(4) / args.scale)))
  

    if args.roi: 
        shade = 2 * np.ones((int(vid.get(4) / args.scale), int(vid.get(3) / args.scale))).astype("uint8")
        shade[int(YMIN/args.scale):int(YMAX/args.scale),int(XMIN/args.scale):int(XMAX/args.scale)] -= 1

    nframe = 0

    
    detected = []

    while True:
  
        ret, frame = vid.read()
  
        nframe += 1

        print(nframe, end = " ", flush = True)

        if not ret: break
        if args.frames and nframe > args.frames: break

        roi = frame if not args.roi else frame[YMIN:YMAX, XMIN:XMAX]

        scaled = cv2.resize(frame, None, fx = 1 / args.scale, fy = 1 / args.scale, interpolation = cv2.INTER_AREA)
        if args.roi: scaled = (scaled / shade[:,:,np.newaxis]).astype("uint8")
        
        # cv2.rectangle(scaled, tuple([int(XMIN // args.scale), int(YMIN // args.scale)]), tuple([int(XMAX // args.scale), int(YMAX // args.scale)]), (255, 255, 255), 2)


        img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
  
        # draw = ImageDraw.Draw(img)
  
        # Run inference.
        ans = engine.detect_with_image(img, threshold = args.thresh, keep_aspect_ratio=args.keep_aspect_ratio, relative_coord=False, top_k = args.k)
  
        # Save result.
        if args.verbose: print('=========================================')
  
        XY = []
        for obj in ans:
  
            label = None
            if labels is not None and len(args.categs):

                label = labels[obj.label_id]
                if label not in args.categs: continue
  

            if label is not None and args.verbose:
                print(labels[obj.label_id] + ",", end = " ")
  
            # Draw a rectangle.
            box = obj.bounding_box.flatten()
            # draw.rectangle(box, outline = colors[args.categs.index(label)] if len(args.categs) else "red", width = 10)
            # print('box = ', box)
            
            color = colors[args.categs.index(label)] if len(args.categs) else (0, 0, 255)


            if args.roi:

                box[0] += XMIN
                box[2] += XMIN
                box[1] += YMIN
                box[3] += YMIN

            draw_box = (box / args.scale).astype(int)
            cv2.rectangle(scaled, tuple(draw_box[:2]), tuple(draw_box[2:]), color, 2)

            if args.verbose: 
                print('conf. = ', obj.score)
                print('-----------------------------------------')

            XY.append((0.5 * (box[0] + box[2]) / args.scale, box[1] / args.scale))
            detected.append([nframe, label, obj.score, box[0], box[1], box[2], box[3]])


        tracker.update(XY)
        tracker.draw(scaled, kalman_cov = args.kalman)
  

        if args.view:
        
            cv2.imshow("view", scaled)
            if (cv2.waitKey(1) & 0xff) == 27: break

        if out is not None: out.write(scaled)
    
        if args.verbose: print('Recorded frame {}'.format(nframe))
  
    if out is not None:

        out.release()

        df = pd.DataFrame(detected, columns = ["frame", "label", "conf", "xmin", "ymin", "xmax", "ymax"])
        df["x"] = 0.5 * (df.xmin + df.xmax)
        df["y"] = 0.5 * (df.ymin + df.ymax)

        df.to_csv(args.input.split(".")[0] + ".csv", index = False)

        tracker.write(args.input.split(".")[0] + "_tracker.csv")


if __name__ == '__main__': main()




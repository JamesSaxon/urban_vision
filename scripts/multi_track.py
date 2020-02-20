#!/usr/bin/env python 

import threading
from time import sleep
import queue
import datetime

from itertools import islice
from collections import deque

import cv2
import os, sys

import argparse

import tracker, detector as det, triangulate as tri

from finding_colored_balls import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, default = "img/")
parser.add_argument("--cams", default = ["E", "W"], nargs = "+")
parser.add_argument("--process", default = False, action = 'store_true')
parser.add_argument("--scale", default = 4, type = float)

parser.add_argument("--categs", default = [], nargs = "+")
parser.add_argument("-k", default = 5, type = int)
parser.add_argument("--max_missing", default = 5, type = int)
parser.add_argument("--max_distance", default = 0.125, type = float)

parser.add_argument("--colors", default = [], nargs = "+")
parser.add_argument("--thresh", default = 0.4, type = float)
parser.add_argument("--max_size", default = 0.02, type = float)
parser.add_argument("--history", default = 1000, type = int)

parser.add_argument("--view", default = False, action = 'store_true')
parser.add_argument("--kalman", default = 0, type = int, help = "Scale of Kalman error covariance in pixels.")
parser.add_argument("--contrail", default = 50, type = int)

parser.add_argument("--triangulate", default = False, action = 'store_true')

args     = parser.parse_args()
# OFILE    = args.file
CAMS     = args.cams
SCALE    = args.scale
HISTORY  = args.history
COLORS   = args.colors
CATEGS   = args.categs
THRESH   = args.thresh
MAX_SIZE = args.max_size


def process(q, n):

    name = threading.currentThread().getName()
    print("created", threading.currentThread().getName())

    detector = det.Detector(categs = CATEGS, thresh = THRESH, k = args.k, verbose = False)

    bkd = cv2.createBackgroundSubtractorMOG2(history = HISTORY, varThreshold = 8, detectShadows = False)

    while not all(COMPLETE.values()):

        try: img, cam = q.get(timeout = 1)
        except: continue

        img = cv2.resize(img, None, fx = 1 / SCALE, fy = 1 / SCALE, interpolation = cv2.INTER_AREA)

        bkd_mask = bkd.apply(img)
        bkd_mask = cv2.erode(bkd_mask, None, iterations = 1)

        fg = cv2.bitwise_or(img, img, mask = bkd_mask)
  
        colors = []
        positions = []
        for color in COLORS:

            balls = get_ball_positions(fg, color = color, 
                                       thresh = THRESH, max_area = MAX_SIZE * img.shape[0] * img.shape[1])

            for ball in balls:

                colors.append(ball_colors_bgr[color])
                positions.append(tuple(ball))

        if CATEGS:
            
            xy, img = detector.detect(img, return_image = True)

            for i_xy in xy:

                colors.append((255, 255, 255))
                positions.append(i_xy)
            

        if len(colors):

            XY[cam] = np.array(positions) * SCALE
            COL[cam] = colors

        else:

            XY[cam] = None
            COL[cam] = None

        tracker[cam].update(np.array(positions) * SCALE, colors = colors)
        img = tracker[cam].draw(img, scale = SCALE, depth = args.contrail, kalman_cov = args.kalman)

        IMG[cam] = img

        q.task_done()


def producer(qprod, qproc):

    print("created", threading.currentThread().getName())

    name = threading.currentThread().getName()

    cam = qprod.get(timeout = 1)

    stream = cap[cam]

    while RECORD:
  
        valid, img = stream.read()
        if not valid: continue

        if qproc.qsize() < 20:

            qproc.put((img, cam))
        

    qproc.join()
    qprod.task_done()

    # out[cam].release()

    COMPLETE[cam] = True

    print("Completed", cam, stream)


addr = {"E" : "rtsp://jsaxon:iSpy53rd@192.168.1.64/1",
        "W" : "rtsp://jsaxon:iSpy53rd@192.168.1.65/1",
        "L" : 0, "M" : 1}

cfile = {"E" : "Ei", "W" : "Wi", "L" : "L", "M" : "M"}

cap = {k : cv2.VideoCapture(addr[k]) for k in CAMS}

##  os.makedirs(OFILE, exist_ok=True)
##  out = {k : cv2.VideoWriter(OFILE + '/{}.mp4'.format(k), # mkv
##                             cv2.VideoWriter_fourcc(*'mp4v'), 30, # X264
##                             (int(cap[k].get(3)) // SCALE, int(cap[k].get(4)) // SCALE))
##         for k in CAMS}

IMG = {k : None for k in CAMS}
XY  = {k : None for k in CAMS}
COL = {k : None for k in CAMS}

RECORD = True
COMPLETE = {k : False for k in CAMS}


if __name__ == '__main__':

    threads = []
    qproc = {}

    for i in [0]: # CAMS
        qproc[i] = queue.Queue(20)
        threads.append(threading.Thread(name = "Processor {}".format(i), target=process, args=(qproc[i], i)))
        threads[-1].start()

    qprod = queue.Queue()
    for i in CAMS:
        threads.append(threading.Thread(name = "Capture {}".format(i), target=producer, args=(qprod, qproc[0])))
        threads[-1].start()

    for i in CAMS: qprod.put(i)

    print("HELLO!!")

    tracker = {k : tracker.Tracker(max_missing = args.max_missing,
                                   max_distance = args.max_distance * cap[k].get(3) / SCALE,
                                   contrail = args.contrail) 
               for k in CAMS}

    triangulate = None
    if args.triangulate and len(CAMS) > 0:
        triangulate = tri.Triangulate(cfile[CAMS[0]], cfile[CAMS[1]],
                                      xmin = -3, xmax = 7, ymin = -4, ymax = 5)

        CAM1, CAM2 = CAMS[0], CAMS[1]

    obs3d = []

    t_new    = datetime.datetime.now()
    t_update = datetime.datetime.now()

    drop_wait = 0

    while True:

        key = (cv2.waitKey(30) & 0xff)
        if key == ord("q"): RECORD = False

        if args.view:

            for k in CAMS:

                if IMG[k] is not None:

                    img = cv2.flip(IMG[k], 1)
                    cv2.imshow(k, img)


        if triangulate is not None:

            t = datetime.datetime.now()
            dt = t - t_new
            
            if dt.total_seconds() > drop_wait:

                triangulate.triangulate(tracker[CAM1].predict_current_locations(),
                                        tracker[CAM2].predict_current_locations(), 
                                        max_reproj_error = 400 / SCALE)
                triangulate.plot()

                t_new = t

                drop_wait = np.random.rand()

            t = datetime.datetime.now()
            dt = t - t_update
            t_update = t

            triangulate.update(dt.total_seconds())
            triangulate.save()

            sleep(0.1)



        if not RECORD: break


    for t in threads:
        print(t)
        t.join()

    np.save("3d_points", np.array(obs3d))

    print("complete ::")
    print("prod: {}".format(qprod.qsize()))
    for k, v in qproc.items():
        print("proc {}: {}".format(k, v.qsize()))
    
    sys.exit()



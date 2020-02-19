#!/usr/bin/env python 

import threading
import time
import queue
import datetime

from itertools import islice
from collections import deque

import cv2
import os, sys

import argparse

import tracker

from finding_colored_balls import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, default = "img/")
parser.add_argument("--cams", default = ["E", "W"], nargs = "+")
parser.add_argument("--process", default = False, action = 'store_true')
parser.add_argument("--scale", default = 4, type = float)
parser.add_argument("--history", default = 1000, type = int)
parser.add_argument("--contrail", default = 50, type = int)
parser.add_argument("--colors", default = ["red"], nargs = "+")
parser.add_argument("--thresh", default = 40, type = int)
parser.add_argument("--max_size", default = 0.02, type = float)

args     = parser.parse_args()
# OFILE    = args.file
CAMS     = args.cams
SCALE    = args.scale
HISTORY  = args.history
CONTRAIL = args.contrail
COLORS   = args.colors
THRESH   = args.thresh
MAX_SIZE = args.max_size


def process(q, n):

    name = threading.currentThread().getName()
    print("created", threading.currentThread().getName())

    bkd = cv2.createBackgroundSubtractorMOG2(history = HISTORY, varThreshold = 8, detectShadows = False)

    nimg = 0
    cam = ""
    while not (cam and COMPLETE[cam]):

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

        if len(colors):

            XY[cam] = positions
            COL[cam] = colors

        else:

            XY[cam] = None
            COL[cam] = None

        IMG[cam] = img

        nimg += 1

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

cfile = {"E" : "Es", "W" : "Ws", "L" : "L", "M" : "M"}

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

    for i in CAMS:
        qproc[i] = queue.Queue(20)
        threads.append(threading.Thread(name = "Processor {}".format(i), target=process, args=(qproc[i], i)))
        threads[-1].start()

    qprod = queue.Queue()
    for i in CAMS:
        threads.append(threading.Thread(name = "Capture {}".format(i), target=producer, args=(qprod, qproc[i])))
        threads[-1].start()

    for i in CAMS: qprod.put(i)

    print("HELLO!!")

    pos = {k : deque() for k in CAMS}
    col = {k : deque() for k in CAMS}

    tracker = {k : tracker.Tracker() for k in CAMS}

    obs3d = []

    while True:

        key = (cv2.waitKey(30) & 0xff)
        if key == ord("q"): RECORD = False
        if key == ord("c"):
            for k in CAMS:
                pos[k].clear()
                col[k].clear()


        for k in CAMS:

            if IMG[k] is not None:
                
                tracker[k].update(XY[k], colors = COL[k])

                img = tracker[k].draw(IMG[k], depth = CONTRAIL)
                img = cv2.flip(img, 1)
                cv2.imshow(k, img)

            if len(CAMS) > 1:
                
                c1, c2 = CAMS[:2]

                if XY[c1] is not None and XY[c2] is not None:

                    pts1 = SCALE * np.array(XY[c1][:1]).T
                    pts2 = SCALE * np.array(XY[c2][:1]).T

                    P1 = np.column_stack([np.eye(3), np.zeros(3)]) # webcam is origin
                    P2 = np.load("P.npy")

                    K1 = np.load('./calib/{}/K.npy'.format(cfile[c1]))
                    K2 = np.load('./calib/{}/K.npy'.format(cfile[c2]))

                    dist1 = np.load('./calib/{}/dist.npy'.format(cfile[c1]))
                    dist2 = np.load('./calib/{}/dist.npy'.format(cfile[c2]))
                    
                    # pts1_norm = cv2.undistortPoints(pts1, cameraMatrix = K1, distCoeffs = dist1)
                    # pts2_norm = cv2.undistortPoints(pts2, cameraMatrix = K2, distCoeffs = dist2)
                    # points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)

                    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
                    
                    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1,3)
        
                    print(points_3d)

                    obs3d.append(points_3d.ravel())

                
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



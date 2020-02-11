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
parser.add_argument("--max_size", default = 40000, type = int)

args     = parser.parse_args()
# OFILE    = args.file
CAMS     = args.cams
SCALE    = args.scale
HISTORY  = args.history
CONTRAIL = args.contrail
COLORS   = args.colors
THRESH   = args.thresh
MAX_SIZE = args.max_size



t0 = datetime.datetime.now()
def dt_str():

  dt = datetime.datetime.now() - t0
  return "{:07.3f}".format(dt.total_seconds())


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
        img = cv2.flip(img, 1)

        bkd_mask = bkd.apply(img)
        bkd_mask = cv2.erode(bkd_mask, None, iterations = 1)

        fg = cv2.bitwise_or(img, img, mask = bkd_mask)
  
        colors = []
        positions = []
        for color in COLORS:

            balls = get_ball_positions(fg, color = color, 
                                       thresh = THRESH, max_area = MAX_SIZE / SCALE / SCALE)

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

            label = dt_str() + "_" + cam
            qproc.put((img, cam))
        

    qproc.join()
    qprod.task_done()

    # out[cam].release()

    COMPLETE[cam] = True

    print("Completed", cam, stream)


addr = {"E" : "rtsp://jsaxon:iSpy53rd@192.168.1.64/1",
        "W" : "rtsp://jsaxon:iSpy53rd@192.168.1.65/1",
        "C" : 0}

cap = {k : cv2.VideoCapture(addr[k]) for k in CAMS}

cup = {"E" : (int(584*2.5), int(215*2.5)), 
       "W" : (int(495*2.5), int(350*2.5))}

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

    tracker = tracker.Tracker()

    while True:

        key = (cv2.waitKey(30) & 0xff)
        if key == ord("q"): RECORD = False
        if key == ord("c"):
            for k in CAMS:
                pos[k].clear()
                col[k].clear()


        for k in CAMS:

            if IMG[k] is not None:
                
                tracker.update(XY[k], colors = COL[k])

                img = tracker.draw(IMG[k], depth = CONTRAIL)
                cv2.imshow(k, img)
                
        if not RECORD: break


    for t in threads:
        print(t)
        t.join()


    print("complete ::")
    print("prod: {}".format(qprod.qsize()))
    for k, v in qproc.items():
        print("proc {}: {}".format(k, v.qsize()))
    
    sys.exit()



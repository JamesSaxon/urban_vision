#!/Users/jsaxon/anaconda/envs/cv/bin/python

import threading
from time import sleep
import queue
import datetime

import cv2
import os, sys

import argparse

from finding_colored_balls import *

parser = argparse.ArgumentParser()
parser.add_argument("--cam1", type = str, required = True)
parser.add_argument("--cam2", type = str, required = True)

parser.add_argument("--delay", type = float, required = True)

parser.add_argument("--scale", default = 1, type = int)
parser.add_argument("--images", default = False, action = "store_true")
parser.add_argument("--nimg", default = 10, type = int)
parser.add_argument("--view", default = False, action = "store_true")
parser.add_argument("--color", default = "red", type = str)

parser.add_argument("--thresh", default = 40, type = int)
parser.add_argument("--max_size", default = 0.02, type = float)

parser.add_argument("--proj_file", default = "P", type = str)

args   = parser.parse_args()
CAM1   = args.cam1
CAM2   = args.cam2
CAMS   = [CAM1, CAM2]

DELAY    = args.delay
SCALE    = args.scale
IMAGES   = args.images
MAX_IMG  = args.nimg
VIEW     = args.view
COLOR    = args.color
MAX_SIZE = args.max_size
HISTORY  = 200
BURN_IN  = 5
THRESH   = 40

OUTPUT = args.proj_file


def process(q, n):

    name = threading.currentThread().getName()
    print("created", threading.currentThread().getName())

    bkd = cv2.createBackgroundSubtractorMOG2(history = HISTORY, varThreshold = 8, detectShadows = False)

    cam = ""
    while not (cam and COMPLETE[cam]):


        try: 
            img, cam, burn, nimg = q.get(timeout = 1)
        except: continue

        img = cv2.resize(img, None, fx = 1 / SCALE, fy = 1 / SCALE, interpolation = cv2.INTER_AREA)

        ##  bkd_mask = bkd.apply(img)

        ##  bkd_mask = cv2.erode(bkd_mask, None, iterations = 1)

        ##  fg = cv2.bitwise_or(img, img, mask = bkd_mask)
        
        balls = get_ball_positions(img, color = COLOR, thresh = THRESH, erode = 4,
                                   max_area = MAX_SIZE * img.shape[0] * img.shape[1])
        # fg = cv2.bitwise_or(img, img, mask = mask)

        # IMG[cam] = fg

        if len(balls): cv2.circle(img, tuple(balls[0]), 50, (0, 255, 0), 3)

        IMG[cam] = img

        if IMAGES and not burn: 
            print(image_name.format(cfile[cam], nimg))
            cv2.imwrite(image_name.format(cfile[cam], nimg), img)

        if not burn:
            if len(balls):
                XY[cam].append(tuple(SCALE * balls[0]))

            else: 
                XY[cam].append(tuple([np.nan, np.nan]))

        q.task_done()


def producer(qprod, qproc):

    print("created", threading.currentThread().getName())

    name = threading.currentThread().getName()

    cam = qprod.get(timeout = 1)

    while NIMG[cam] < MAX_IMG:
  
        valid, img = CAP[cam].read()

        if not valid: continue

        if RECORD[cam]: 

            NIMG[cam] += 1
            RECORD[cam] = False

            qproc.put((img, cam, NIMG[cam] < 0, NIMG[cam]))

    qproc.join()
    COMPLETE[cam] = True
    qprod.task_done()

    print("Completed", cam)


addr = {"E" : "rtsp://jsaxon:iSpy53rd@192.168.1.64/1",
        "W" : "rtsp://jsaxon:iSpy53rd@192.168.1.65/1",
        "0" : 0, "1" : 1}

cfile = {"E" : "Es", "W" : "Ws", "0" : "L", "1" : "M"}

CAP = {k : cv2.VideoCapture(addr[k]) for k in CAMS}


image_name = "calib/{}/points/{:03d}.jpg"

for k in CAMS:
    os.makedirs("calib/{}/points/".format(cfile[k]), exist_ok=True)

    
print(CAMS)
for k in CAMS: print(k, CAP[k].get(cv2.CAP_PROP_FPS), "fps")

IMG = {k : None for k in CAMS}
XY  = {k : [] for k in CAMS}

STOP = False

NIMG     = {k : -BURN_IN for k in CAMS}
RECORD   = {k : False for k in CAMS}
COMPLETE = {k : False for k in CAMS}

if __name__ == '__main__':

    threads = []
    qproc = {}

    for i in CAMS:
        qproc[i] = queue.Queue(20)
        threads.append(threading.Thread(name = "Processor {}".format(i), target = process, args=(qproc[i], i)))
        threads[-1].start()

    qprod = queue.Queue()
    for i in CAMS:
        threads.append(threading.Thread(name = "Capture {}".format(i), target=producer, args=(qprod, qproc[i])))
        threads[-1].start()

    for i in CAMS: qprod.put(i)

    print("HELLO!!")

    if VIEW:

      while not all(COMPLETE.values()):

          sleep(DELAY)

          for k in CAMS:

              RECORD[k] = True

              if IMG[k] is None: continue

              if VIEW: cv2.imshow("img" + k, IMG[k])

          key = (cv2.waitKey(30) & 0xff)

          if key == ord("q"): STOP = True


    for t in threads:
        print(t)
        t.join()


    for k in CAMS: 

        XY[k] = np.array(XY[k])
        print(k, XY[k])
  
    mask = np.logical_or(np.isnan(XY[CAM1]), np.isnan(XY[CAM2]))
    for k in CAMS: 

        XY[k] = np.ma.array(XY[k], mask = mask)
        print(k, XY[k])

    F, mask = cv2.findFundamentalMat(XY[CAM1], XY[CAM2])

    K1 = np.load('./calib/{}/K.npy'.format(cfile[CAM1]))
    K2 = np.load('./calib/{}/K.npy'.format(cfile[CAM2]))

    E = np.dot(K2.T, np.dot(F, K1))  # K2'*F*K1

    # I THINK that K1 is only used for the "chirality check," i.e.,
    # that the 3D points have positive depth / are in front of the cam.
    # It is just decomposeEssentialMat(E) but with checks.
    # So perhaps it is OK that K1 is NOT shared for CAM2???
    _, R, t, _ = cv2.recoverPose(E, XY[CAM1], XY[CAM2], K1, mask = mask)

    proj = np.hstack((R, t))

    print(R)
    print(t)
    print(proj)

    np.save(OUTPUT, proj)

    print("complete ::")
    print("prod: {}".format(qprod.qsize()))
    for k, v in qproc.items():
        print("proc {}: {}".format(k, v.qsize()))
    
    sys.exit()



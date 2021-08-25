#!/Users/jsaxon/anaconda/envs/cv/bin/python

import threading
import time
import queue
import datetime

import cv2
import os, sys

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, default = "img/")
parser.add_argument('--nimg', type=int, default = 1000)
parser.add_argument("--cams", default = ["E", "W"], nargs = "+")
parser.add_argument("--process", default = False, action = 'store_true')
parser.add_argument("--scale", default = 4, type = int)

args  = parser.parse_args()
OFILE   = args.file
NIMG    = args.nimg
CAMS    = args.cams
SCALE   = args.scale

os.makedirs(OFILE, exist_ok=True)


t0 = datetime.datetime.now()
def dt_str():

  dt = datetime.datetime.now() - t0
  return "{:07.3f}".format(dt.total_seconds())


def process(q, n):

    name = threading.currentThread().getName()
    print("created", threading.currentThread().getName())

    cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    print("loaded cascade")

    cam = ""
    while not (cam and COMPLETE[cam]):

        try: img, cam = q.get(timeout = 1)
        except: continue

        img = cv2.resize(img, None, fx = 1 / SCALE, fy = 1 / SCALE, interpolation = cv2.INTER_AREA)

        out[cam].write(img)

        q.task_done()
        # print("processed {} // qproc size now = {}".format(cam, q.qsize()))


def producer(qprod, qproc):

    print("created", threading.currentThread().getName())

    name = threading.currentThread().getName()

    cam = qprod.get(timeout = 1)

    stream = cap[cam]

    for nimg in range(NIMG):
  
        valid, img = stream.read()
        if not valid: continue

        if qproc.qsize() < 20:

            label = dt_str() + "_" + cam
            print(cam, nimg)
            qproc.put((img, cam))
        

    qproc.join()
    qprod.task_done()

    out[cam].release()

    COMPLETE[cam] = True

    print("Completed", stream)


addr = {"E" : "rtsp://jsaxon:iSpy53rd@192.168.1.64/1",
        "W" : "rtsp://jsaxon:iSpy53rd@192.168.1.65/1"}

cap = {k : cv2.VideoCapture(addr[k]) for k in CAMS}

out = {k : cv2.VideoWriter(OFILE + '/{}.mp4'.format(k), # mkv
                           cv2.VideoWriter_fourcc(*'mp4v'), 30, # X264
                           (int(cap[k].get(3)) // SCALE, int(cap[k].get(4)) // SCALE))
       for k in CAMS}

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

    for t in threads:
        print(t)
        t.join()

    print("complete ::")
    print("prod: {}".format(qprod.qsize()))
    for k, v in qproc.items():
        print("proc {}: {}".format(k, v.qsize()))
    
    sys.exit()



#!/Users/jsaxon/anaconda/envs/cv/bin/python

import threading
import time
import queue
import datetime

import cv2
import os

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, default = "img/")
parser.add_argument('--nimg', type=int, default = 1000)
parser.add_argument("--cams", default = ["E", "W"], nargs = "+")
parser.add_argument("--process", default = False, action = 'store_true')

args  = parser.parse_args()
OFILE   = args.file
NIMG    = args.nimg
CAMS    = args.cams

os.makedirs(OFILE, exist_ok=True)


THREADS_COMPLETE = False

t0 = datetime.datetime.now()
def dt_str():

  dt = datetime.datetime.now() - t0
  return "{:07.3f}".format(dt.total_seconds())


def process(q, n):

    name = threading.currentThread().getName()
    print("created", threading.currentThread().getName())

    cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    print("loaded cascade")

    scale = 0.5

    while not THREADS_COMPLETE:

        img, label = q.get()

        img = cv2.resize(img, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cascade_obj = cascade.detectMultiScale(gray)
  
        # for (x,y,w,h) in cascade_obj:
        #     cv2.rectangle(img, (x, y), ((x+w), (y+h)), (0, 255, 255), 2)

        cv2.imwrite('{}/{}.jpg'.format(OFILE, label), img)

        q.task_done()
        print("processed {} // qproc size now = {}".format(label, q.qsize()))


def producer(qprod, qproc):

    print("created", threading.currentThread().getName())

    while not THREADS_COMPLETE:

      name = threading.currentThread().getName()

      cam = qprod.get()
      stream = cap[cam]

      print(name, stream, NIMG)

      for i in range(NIMG):
  
          # label = "{:03d}".format(i) + "_" + "EW"[nimg]


          valid, img = stream.read()
          if not valid: continue
          # print("updated", label)
          # cv2.imshow('img_{}'.format(nimg), img)
          # cv2.imwrite('img/{}_{}.jpg'.format(dt_str(), "EW"[nimg]), img)

          if qproc.qsize() < 20:
            label = dt_str() + "_" + cam
            qproc.put((img, label))
          

      qproc.join()
      qprod.task_done()
      print("Completed", stream)


addr = {"E" : "rtsp://jsaxon:iSpy53rd@192.168.1.64/1",
        "W" : "rtsp://jsaxon:iSpy53rd@192.168.1.65/1"}

cap = {k : cv2.VideoCapture(addr[k]) for k in CAMS}

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

    print(cap)
    for i in CAMS: qprod.put(i)

    print("HELLO!!")
    time.sleep(10)

    # while qprod.qsize() or qproc.qsize():
    #   print("still working", qprod.qsize(), qproc.qsize())
    #   time.sleep(1)

    # print("now", qprod.qsize(), qproc.qsize())

    # THREADS_COMPLETE = True
    # time.sleep(0.5)
    # for t in threads: t.join()



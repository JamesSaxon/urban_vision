#!/Users/jsaxon/anaconda/envs/cv/bin/python

import cv2 
import os, sys, re

video = "lsd_cars/lsd_cars.mov"
opath = re.sub(r".*\/(.*).mov", r"\1", video)
  
cam = cv2.VideoCapture(video)
os.makedirs(opath, exist_ok=True) 
  
SCALE = 4

nframe = 0

while True: 

    # reading from frame 
    ret,frame = cam.read() 
    nframe += 1

    if ret: 

        if (cv2.waitKey(30) & 0xff) == 27: break


        frame = cv2.resize(frame, None, fx = 1 / SCALE, fy = 1 / SCALE, interpolation = cv2.INTER_AREA)

        cv2.imshow('frame', frame)
  
    else: break
  
cam.release() 
cv2.destroyAllWindows() 


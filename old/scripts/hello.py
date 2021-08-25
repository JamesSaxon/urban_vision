#!/Users/jsaxon/anaconda/envs/cv/bin/python

import cv2 
import numpy as np
import glob, re, sys

from finding_colored_balls import *

color = "circles"

start, stop = "013.", "037"

for cam in "EW":

  record = False
  ctrs = []
  for fi, f in enumerate(glob.glob("img/hello_3D/raw/*_{}.jpg".format(cam))):
  
      print(fi, f)
      if start in f: record = True
      if stop  in f: record = False
  
      if record:
  
        xy = find_balls_in_image(f, color = color, thresh = 40)
        ctrs.append(tuple(int(x) for x in np.array(xy).ravel()))
  
      img = cv2.imread(f)
  
      if len(ctrs) > 1:

        X0, X1 = None, None
        for X1 in ctrs:

          if not len(X1): continue

          if X0 is not None:
              img = cv2.line(img, X0, X1, ball_colors_bgr[color], 3)

          X0 = X1

  
      cv2.imwrite("img/hello_3D/frames/{}_{:03d}.jpg".format(cam, fi), img)
      # print(fi, end = " ", flush = True)



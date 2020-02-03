#!/Users/jsaxon/anaconda/envs/cv/bin/python

import numpy as np
import cv2 
import glob

import imutils

from time import sleep, time

import re, sys

ball_colors_bgr = {"red" : (35, 35, 160),
                   "circles" : (35, 0, 150),
                   "yellow" : (48, 145, 170), 
                   "green" : (40, 100, 40),
                   "hello" : (45, 105, 35),
                   "blue" : (130, 30, 15),
                   "magenta" : (90, 45, 155)}

def find_balls_in_image(fname, color = "red", thresh = 0.2, debug = False):

    if debug: print(fname, color, thresh, debug)
    lab = np.array(ball_colors_bgr[color], dtype="uint8").reshape(1, 1, 3)
    lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)[0,0,:].astype(int)

    img = cv2.imread(fname)
    blur = cv2.GaussianBlur(img, (11, 11), 0)
    img_lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB)
    
    lab_diff = np.subtract(img_lab, lab.astype(int))
    lab_diff = np.sqrt(np.sum(lab_diff ** 2, axis = 2))
    
    diff_norm = lab_diff / 100
    diff_norm[diff_norm > 1] = 1
    
    diff = np.array(diff_norm * 255, dtype = np.uint8)
    mask = np.array((lab_diff < thresh) * 255, dtype = np.uint8)
    mask = cv2.erode(mask, None, iterations = 1)
    mask = cv2.dilate(mask, None, iterations = 1)
    

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    for c in contours:
    
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 25: continue

        cv2.circle(img,  (int(x), int(y)), int(radius) + 10, ball_colors_bgr[color], 3)
        cv2.circle(diff, (int(x), int(y)), int(radius) + 10, ball_colors_bgr[color], 3)
        cv2.circle(mask, (int(x), int(y)), int(radius) + 10, ball_colors_bgr[color], 3)
        
        balls.append([x, y])


    if debug:
        cv2.imwrite(fname.replace("raw/", "{}/".format(color)), img)
        cv2.imwrite(fname.replace("raw/", "diff/"), diff)
        cv2.imwrite(fname.replace("raw/", "mask/"), mask)

    return balls

##  find_balls_in_image("Ei.jpg", color = "red", thresh = 0.1, debug = True)
##  find_balls_in_image("Wi.jpg", color = "red", thresh = 0.1, debug = True)


##  for f in glob.glob("img/hello_3D/raw/*[EW].jpg"):
##  
##    find_balls_in_image(f, color = "circles", thresh = 40, debug = True)

##  for f in glob.glob("calib/?i/ref_loc/?_?_?.jpg"):
##  
##    find_balls_in_image(f, color = "hello", thresh = 0.1, debug = True)

##  for v in ball_colors_bgr:
##    find_balls_in_image("picture_A_0_cube.jpg", color = v, thresh = 0.1)
##  
##  
##  for cam in "EW":
##    find_balls_in_image("calib/{}i/ref_loc/3_3_0.jpg".format(cam), 
##                        color = "green", thresh = 0.2)


#!/Users/jsaxon/anaconda/envs/cv/bin/python

import numpy as np
import cv2 
import glob

#import imutils

from time import sleep, time

import re, sys

ball_colors_bgr = {# "red" : (20, 45, 180),
                   "red" : (60, 15, 150),
                   "orange" : (30, 80, 190),
                   "yellow" : (0, 190, 190), 
                   "green" : (50, 90, 25),
                   "blue" : (130, 30, 15),
                   "magenta" : (60, 45, 180)}

ball_colors_lab = {k : cv2.cvtColor(np.array(v, dtype="uint8").reshape(1, 1, 3), 
                                    cv2.COLOR_RGB2LAB)[0,0,:].astype(int)
                   for k, v in ball_colors_bgr.items()}


def get_ball_color_mog(roi, bkd_color, blur = 5, thresh = 20):

    mask = bkd_color.apply(roi, learningRate = 0)
    mask = cv2.erode(mask, None, iterations = 2)

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cv2.bitwise_or(roi, roi, mask = mask)


    balls = []
    detections = []
    for c in contours:

        area = cv2.contourArea(c)

        # print(color, area)
        if area < 25:  continue
        if area > 800: continue

        for color, lab in ball_colors_lab.items():

            mask = np.zeros(roi.shape[:2], np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask3d = np.broadcast_to(np.expand_dims(mask > 0, axis = 2), roi_lab.shape)
            mean_color = roi_lab[mask > 0].mean(axis = 0)
            # print(mean_color)

            color_distance = np.sqrt(np.sum(np.subtract(mean_color, lab.astype(int))**2))

            if color == "yellow" and color_distance > 15: continue

            detections.append([color, mean_color, color_distance, area])


    if len(detections):

        # print("==========================")
        # for d in detections: print(d)

        ## Smallest color distance...
        detections.sort(key = lambda x: x[2])
        detection = detections[0]
        
        lab_color = np.array(detection[1], dtype="uint8").reshape(1, 1, 3)
        bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2RGB)[0,0,:]
        bgr_color = tuple(bgr_color.ravel())
        
        print(detection, bgr_color)
        # print(detection[0], end = " ", flush = True)

        return bgr_color

    return None


 
def get_ball_color(img, xy, size = 80, blur = 5, thresh = 20):

    y, x = xy

    roi = img[x-size//2:x+size//2, y-size//2:y+size//2]

    detections = []
    for color, value in ball_colors_bgr.items():

        lab = np.array(value, dtype="uint8").reshape(1, 1, 3)
        lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)[0,0,:].astype(int)

        img_blur = cv2.GaussianBlur(roi, (blur, blur), 0)
        img_lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)

        lab_diff = np.subtract(img_lab, lab.astype(int))
        lab_diff = np.sqrt(np.sum(lab_diff ** 2, axis = 2))
        
        mask = np.array((lab_diff < thresh) * 255, dtype = np.uint8)
        mask = cv2.dilate(mask, None, iterations = 2)
        mask = cv2.erode(mask, None, iterations = 2)

        # The [-2:] is to deal with differences in cv2 versions.
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        balls = []
        for c in contours:

            area = cv2.contourArea(c)

            # print(color, area)
            if area < 50:  continue
            if area > 700: continue

            mask = np.zeros(roi.shape[:2], np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask3d = np.broadcast_to(np.expand_dims(mask > 0, axis = 2), img_lab.shape)
            mean_color = img_lab[mask > 0].mean(axis = 0)
            # print(mean_color)

            color_distance = np.sqrt(np.sum(np.subtract(mean_color, lab.astype(int))**2))

            if color == "yellow" and color_distance > 15: continue

            detections.append([color, mean_color, color_distance, area])


    if len(detections):

        # print("==========================")
        # for d in detections: print(d)

        ## Smallest color distance...
        detections.sort(key = lambda x: x[2])
        detection = detections[0]
        
        lab_color = np.array(detection[1], dtype="uint8").reshape(1, 1, 3)
        bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2RGB)[0,0,:]
        bgr_color = tuple(bgr_color.ravel())
        
        # print(detection, bgr_color)
        print(detection[0], end = " ", flush = True)

        return bgr_color

    return None


def get_ball_positions(img, color = "red", blur = 3, thresh = 50, max_area = 150, erode = 4):

    if type(color) is str:
        color = ball_colors_bgr[color]
    
    # print(color, ball_colors_bgr["magenta"])

    lab = np.array(color, dtype="uint8").reshape(1, 1, 3)
    lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)[0,0,:].astype(int)

    img_blur = cv2.GaussianBlur(img, (blur, blur), 0)
    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)

    lab_diff = np.subtract(img_lab, lab.astype(int))
    lab_diff = np.sqrt(np.sum(lab_diff ** 2, axis = 2))
    
    mask = np.array((lab_diff < thresh) * 255, dtype = np.uint8)
    mask = cv2.erode(mask, None, iterations = erode)
    mask = cv2.dilate(mask, None, iterations = erode + 2)

    # The [-2:] is to deal with changes is in cv2 versions.
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse = True)

    balls = []
    for c in contours:

        area = cv2.contourArea(c)

        if area > max_area: continue

        M = cv2.moments(c)

        if M["m00"]: 
            balls.append([M["m10"] / M["m00"],
                          M["m01"] / M["m00"]])

    return np.array(balls).astype(int) # , mask



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


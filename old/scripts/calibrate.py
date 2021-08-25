#!/Users/jsaxon/anaconda/envs/cv/bin/python

import os
from sys import exit
import cv2
from time import sleep, time

import multiprocessing

import datetime
import argparse

import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image

camera = "Wi"

#============================================
# Camera calibration
#============================================
#Define size of chessboard target.
chessboard_size = (6,9)


#Define arrays to save detected points
obj_points = [] #3D points in real world space 
img_points = [] #3D points in image plane

#Prepare grid and points to display
objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

#read images
calibration_paths = glob.glob('./calib/{}/chess/*.jpg'.format(camera))
print('./calib/{}/chess/*.jpg'.format(camera))
print(calibration_paths)

#Iterate over images to find intrinsic matrix
for image_path in tqdm(calibration_paths):
  #Load image
  image = cv2.imread(image_path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  if (cv2.waitKey(30) & 0xff) == 27: break
  cv2.imshow("chess", gray_image)

  #find chessboard corners
  ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
  
  if ret == True:
    #define criteria for subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #refine corner location (to subpixel accuracy) based on criteria.
    cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
    obj_points.append(objp)
    img_points.append(corners)

print("Detected boards in {} images.".format(len(img_points)))

#Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_image.shape[::-1], None, None)

#Save parameters into numpy file
np.save("./calib/{}/ret".format(camera), ret)
np.save("./calib/{}/K".format(camera), K)
np.save("./calib/{}/dist".format(camera), dist)
np.save("./calib/{}/rvecs".format(camera), rvecs)
np.save("./calib/{}/tvecs".format(camera), tvecs)




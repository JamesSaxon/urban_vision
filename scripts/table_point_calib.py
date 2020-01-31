#!/Users/jsaxon/anaconda/envs/cv/bin/python

import numpy as np
import cv2 
import glob

import imutils

from time import sleep, time

import re, sys

from finding_colored_balls import find_balls_in_image, ball_colors_bgr

iname = "calib/{}/ref_loc/{:.0f}_{:.0f}_{:.0f}.jpg"

world_coords = {}
for cam in ["Ei", "Wi"]:

    loc = glob.glob("calib/{}/ref_loc/[0-9]_[0-9]_[0-9].jpg".format(cam))
    loc = [re.sub(r".*/([0-9]_[0-9]_[0-9]).jpg", r"\1", v) for v in loc]
    loc = [v.split("_") for v in loc]
    loc = [[float(vi) for vi in v] for v in loc]

    world_coords[cam] = loc

for c1, c2 in [["Ei", "Wi"], ["Wi", "Ei"]]:
    world_coords[c1] = [x for x in world_coords[c1] if x in world_coords[c2]]

for c1, c2 in [["Ei", "Wi"], ["Wi", "Ei"]]:
    world_coords[c1] = np.array(world_coords[c1])

image_coords = {"Ei" : [], "Wi" : []}

rvec, tvec = {}, {}

zero = np.float32([[0,0,0]]).reshape(-1,3)
axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)


for cam, loc_list in world_coords.items():

  # if cam == "Ei": continue

  for loc in loc_list:

    this_iname = iname.format(cam, loc[0], loc[1], loc[2])

    ball_loc = find_balls_in_image(this_iname, color = "hello", thresh = 0.1)

    image_coords[cam].append(ball_loc)

  image_coords[cam] = np.array(image_coords[cam])

  matrix     = np.load('./calib/{}/K.npy'.format(cam))
  distortion = np.load('./calib/{}/dist.npy'.format(cam))

  #  print("objp")
  #  print(world_coords[cam])

  #  print("image")
  #  print(image_coords[cam])
  _, rvecs, tvecs = cv2.solvePnP(world_coords[cam], image_coords[cam], matrix, distortion)

  zero_proj, _ = cv2.projectPoints(zero, rvecs, tvecs, matrix, distortion)
  axis_proj, _ = cv2.projectPoints(axis, rvecs, tvecs, matrix, distortion)
  # print(zero_proj[0].ravel())
  # print(axis_proj[0].ravel())
  # print(axis_proj[1].ravel())
  # print(axis_proj[2].ravel())


  img = cv2.imread("{}.jpg".format(cam))
  
  for pt in image_coords[cam]:
      # print(pt.ravel().astype(int))
      cv2.circle(img, tuple(pt.ravel().astype(int)), 10, (0, 255, 0), -1)

  # print(tuple(zero_proj[0].ravel()))
  img = cv2.line(img, tuple(zero_proj[0].ravel()), tuple(axis_proj[0].ravel()), (255,0,0), 5)
  img = cv2.line(img, tuple(zero_proj[0].ravel()), tuple(axis_proj[1].ravel()), (0,255,0), 5)
  img = cv2.line(img, tuple(zero_proj[0].ravel()), tuple(axis_proj[2].ravel()), (0,0,255), 5)
  # cv2.circle(img, tuple(zero_proj[0].ravel()), 15, (255, 255, 255), 5)

  cv2.imwrite(cam + "_axes.jpg", img)

  np.save("calib/{}/rvec".format(cam), rvecs)
  np.save("calib/{}/tvec".format(cam), tvecs)

  rvec[cam] = rvecs
  tvec[cam] = tvecs



## Now let's see if we can plot those rays!!

# Get a fresh image to draw on.
img = cv2.imread("Wi.jpg")

# Create the rotation matrix from the rvec, and construct R|t as 4x4.
Ei_proj = np.concatenate((cv2.Rodrigues(rvec["Ei"])[0], tvec["Ei"]), axis=1)
Ei_proj = np.concatenate((Ei_proj, np.array([[0, 0, 0, 1]])), axis = 0)

# Invert this to get the camera positions.
Ei_camera_coords = np.array([np.linalg.inv(Ei_proj)[:3,3]])

# Now get the camera intrinsics for Wi.
K = np.load('./calib/Wi/K.npy')
dist = np.load('./calib/Wi/dist.npy')

# use these to project camera Ei's position into the Wi scene.
zero_proj_Wi, _   = cv2.projectPoints(zero, rvec["Wi"], tvec["Wi"], K, dist)
Ei_camera_proj_Wi, _ = cv2.projectPoints(Ei_camera_coords, rvec["Wi"], tvec["Wi"], K, dist)

# Draw the line from here to zero.
img = cv2.line(img, tuple(Wi_zero_proj.ravel().astype(int)), 
                    tuple(Ei_camera_proj.ravel().astype(int)), (255, 255, 255), 3)

cv2.imwrite("Wi_ray.jpg", img)



img = cv2.imread("Ei.jpg")

Wi_proj = np.concatenate((cv2.Rodrigues(rvec["Wi"])[0], tvec["Wi"]), axis=1)
Wi_proj = np.concatenate((Wi_proj, np.array([[0, 0, 0, 1]])), axis = 0)
Wi_camera_coords = np.array([np.linalg.inv(Wi_proj)[:3,3]])

K = np.load('./calib/Ei/K.npy')
dist = np.load('./calib/Ei/dist.npy')

zero_proj_Ei, _   = cv2.projectPoints(zero, rvec["Ei"], tvec["Ei"], K, dist)
Wi_camera_proj_Ei, _ = cv2.projectPoints(Wi_camera_coords, rvec["Ei"], tvec["Ei"], K, dist)

img = cv2.line(img, tuple(zero_proj_Ei.ravel().astype(int)), 
                    tuple(Wi_camera_proj_Ei.ravel().astype(int)), (255, 255, 255), 3)

cv2.imwrite("Ei_ray.jpg", img)



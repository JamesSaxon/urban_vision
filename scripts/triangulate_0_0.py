#!/Users/jsaxon/anaconda/envs/cv/bin/python

import cv2 
import numpy as np
import glob, re, sys

from finding_colored_balls import find_balls_in_image, ball_colors_bgr

E_K    = np.load('./calib/Ei/K.npy')
E_dist = np.load('./calib/Ei/dist.npy')
E_rvec = np.load('./calib/Ei/rvec.npy')
E_tvec = np.load('./calib/Ei/tvec.npy')

W_K    = np.load('./calib/Wi/K.npy')
W_dist = np.load('./calib/Wi/dist.npy')
W_rvec = np.load('./calib/Wi/rvec.npy')
W_tvec = np.load('./calib/Wi/tvec.npy')


E_uv = np.array([find_balls_in_image("Ei.jpg", color = "red", thresh = 0.1)])
W_uv = np.array([find_balls_in_image("Wi.jpg", color = "red", thresh = 0.1)])

points = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]]).reshape(-1,3)
E_pts_reproj, _ = cv2.projectPoints(points, E_rvec, E_tvec, E_K, E_dist)
W_pts_reproj, _ = cv2.projectPoints(points, W_rvec, W_tvec, W_K, W_dist)

E_proj = np.dot(E_K, np.concatenate((cv2.Rodrigues(E_rvec)[0], E_tvec), axis=1))
W_proj = np.dot(W_K, np.concatenate((cv2.Rodrigues(W_rvec)[0], W_tvec), axis=1))


# print(E_uv, W_uv)
red_points = cv2.triangulatePoints(E_proj, W_proj, E_uv, W_uv)
red_points /= red_points[3] # homogeneous to euclidean
print("red point:", red_points[0:3].T.round(1))

reproj_points = cv2.triangulatePoints(E_proj, W_proj, E_pts_reproj, W_pts_reproj)
reproj_points /= reproj_points[3] # homogeneous to euclidean
print("original:")
print(points)
print("reprojected:")
print(reproj_points[0:3].T.round(1))



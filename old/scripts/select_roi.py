#!/usr/bin/env python3

import os
import argparse
import cv2

import pandas as pd

def get_roi(video, csv, dxy, fps, skip): 

    vid = cv2.VideoCapture(video)

    ret, img = vid.read()

    for f in range(fps * skip):
        ret, img = vid.read()
        print(f)

    if not ret:
        print("Video file '{}' is not valid.".format(vinput))
        sys.exit()

    WIDTH  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = max([2, WIDTH / dxy[0], HEIGHT / dxy[1]])
    print(dxy, WIDTH, HEIGHT)
    print(scale)

    img = cv2.resize(img, None, fx = 1 / scale, fy = 1 / scale)

    ROI = cv2.selectROI(img)
    cv2.destroyWindow("ROI selector")

    ROI = [int(x * scale) for x in ROI]

    XMIN, XMAX = ROI[0] / WIDTH,  (ROI[0] + ROI[2]) / WIDTH
    YMIN, YMAX = ROI[1] / HEIGHT, (ROI[1] + ROI[3]) / HEIGHT

    roi_dict = {"file" : video, "xmin" : XMIN, "xmax" : XMAX, "ymin" : YMIN, "ymax" : YMAX}

    roi_df = pd.DataFrame([roi_dict])

    if os.path.exists(csv):

        roi_old_df = pd.read_csv(csv)
        roi_df = pd.concat([roi_old_df, roi_df], axis = 0)
        roi_df.drop_duplicates("file", inplace = True, keep = "last")
        roi_df.reset_index(drop = True, inplace = True)

    roi_df = roi_df.round(2)
    roi_df.sort_values(by = "file", inplace = True)
    roi_df.to_csv(csv, index = False, header = True)

    print(roi_df)

    return ROI




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get and stash the ROI for a file')

    parser.add_argument('--video', type=str, help = "video file name", default = "../../../data/cv/vid/burnham/55/20200921_135220.MOV")
    parser.add_argument('--csv',   type=str, help = "csv to append to", default = "roi_defaults.csv")
    parser.add_argument('--skip',  type=int, help = "seconds of video to skip", default = 20)
    parser.add_argument('--fps',   type=int, help = "true fps", default = 5)
    parser.add_argument("--dxy",   default = [1680, 850], type = int, nargs = 2, help = "screen width, height")

    args  = parser.parse_args()

    get_roi(**vars(args))






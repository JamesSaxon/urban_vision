#!/usr/bin/env python 

import os, sys

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon

import itertools

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--video',    type = str, default = "/home/jsaxon/proj/vid/city/bike_path_210901_173435.mp4")
parser.add_argument('--tracks',   type = str, default = "/home/jsaxon/proj/vid/city/bike_path_210901_173435_tr.csv")
parser.add_argument('--regions',  type = str, default = "../data/turn_boxes/north_side_bike_path.geojson")
parser.add_argument('--counts',   type = str, default = "")
parser.add_argument('--min_det',  type = int, default = 2)
parser.add_argument('--alpha',    type = float, default = 0.1)
parser.add_argument('--no_count', action = "store_true")
parser.add_argument('--scale',    type = float, default = 1.0)
parser.add_argument('--roi_file', type = str, default = "roi.conf")
args  = parser.parse_args()

ALPHA = args.alpha
MIN_DET = args.min_det
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
SCALE = args.scale

if not args.tracks and args.video:
    template = args.video.replace(".mp4", "").replace(".MOV", "")

if not args.regions:

    if args.tracks:
        template = args.tracks.split("/")[-1].replace("_tr.csv", "")
        data_dir = "../data/turn_boxes/"
        args.regions = data_dir + template + ".geojson"

    if args.video:
        template = args.video.split("/")[-1].replace(".mp4", "").replace(".MOV", "")
        data_dir = "../data/turn_boxes/"
        args.regions = data_dir + template + ".geojson"

    if not args.regions:
        print("Must provide regions, somehow!!")
        sys.exit()

ROI = None
if args.roi_file:

    roi_df = pd.read_csv(args.roi_file)

    indices = [si in args.video for si in roi_df.file]

    if any(indices):

        index = indices.index(True)
        ROI = roi_df.loc[index].to_dict()
        print(ROI)


if not args.counts and args.tracks:
    args.counts = args.video.replace("_tr.csv", "_turn_counts.csv")


xpt, ypt, finished = None, None, False
def set_click_location(evt, x, y, flags, params):

    global xpt, ypt, finished

    if evt == cv2.EVENT_LBUTTONDBLCLK:
        finished = True
        xpt, ypt = int(x * SCALE), int(y * SCALE) 

    if evt == cv2.EVENT_LBUTTONDOWN:
        xpt, ypt = int(x * SCALE), int(y * SCALE) 



def build_regions(vfile, rfile):

    if os.path.isfile(rfile):
        regions = gpd.read_file(rfile).set_index("region")
        regions.crs = None
        return regions

    vid = cv2.VideoCapture(vfile)
    ret, orig_frame = vid.read()
    vid.release()

    FRAMEX, FRAMEY = orig_frame.shape[1], orig_frame.shape[0]

    orig_frame_disp = cv2.resize(orig_frame, None,
                                 fx = 1 / SCALE, fy = 1 / SCALE,
                                 interpolation = cv2.INTER_AREA)

    cv2.imshow("image", orig_frame_disp)

    cv2.setMouseCallback("image", set_click_location)

    OK = False
    poly = None

    global xpt, ypt, finished
    cv_polygons, gpd_polygons = [], {}

    while True:

        dec_img = orig_frame.copy()

        for col, pg in zip(COLORS, cv_polygons):
            
            cv2.polylines(dec_img, [pg], True, col, 2)

            overlay_img = dec_img.copy()
            cv2.fillPoly(overlay_img, [pg], col)

            dec_img = ((1 - ALPHA) * dec_img + ALPHA * overlay_img).astype(np.uint8)

        if poly is not None and poly.size > 2:
            
            cv2.polylines(dec_img, [poly], finished, (255, 255, 255), 3)


        key = cv2.waitKey(10)

        if key == ord("o"):
            OK = True
            continue

        if key == ord("k") and OK:
            break

        if key >= 0:
            OK = True

        if key == ord("u"):
            
            if poly is not None:
                if poly.size == 2:
                    poly = None
                else:
                    poly = poly[:-1]

            elif cv_polygons:
                cv_polygons = cv_polygons[:-1]
                gpd_polygons.popitem()

        if key == ord("c"):
            poly = None

        if xpt is not None:

            ## Add the first point
            if poly is None:
                poly = np.array([[xpt, ypt]])
    
            ## Add new, different points.
            elif xpt != poly[-1,0] or ypt != poly[-1,1]:
                poly = np.concatenate((poly, np.array([[xpt, ypt]])))

            ## If it is finished, close the ring.
            if finished:

                poly = np.concatenate((poly, poly[:1,]))

                cv2.polylines(dec_img, [poly], False, (255, 255, 255), 3)

                dec_img_disp = cv2.resize(dec_img, None,
                                          fx = 1 / SCALE, fy = 1 / SCALE,
                                          interpolation = cv2.INTER_AREA)
               
                cv2.imshow("image", dec_img_disp)

                print("Enter a unique label for this polygon:", end = " ", flush = True)

                while True:

                    key = cv2.waitKey(10)

                    if key < 0 or key == 225: continue

                    label = chr(key).upper()
                    if label in gpd_polygons: continue

                    print(label)

                    break

                polygon = Polygon(poly)

                if polygon.exterior.is_ccw:
                    gpd_polygons[label] = polygon
                    cv_polygons.append(poly[::-1])
                    
                else:
                    polygon = Polygon(poly[::-1])
                    gpd_polygons[label] = polygon
                    cv_polygons.append(poly)

                poly = None

            xpt, ypt, finished = None, None, False

        if ROI:
            dec_img = cv2.rectangle(dec_img,
                                    tuple((int(ROI["xmin"] * FRAMEX), int(ROI["ymin"] * FRAMEY))),
                                    tuple((int(ROI["xmax"] * FRAMEX), int(ROI["ymax"] * FRAMEY))),
                                    (255, 255, 255), 3)

        dec_img_disp = cv2.resize(dec_img, None,
                                  fx = 1 / SCALE, fy = 1 / SCALE,
                                  interpolation = cv2.INTER_AREA)
               
        cv2.imshow("image", dec_img_disp)


    if not len(gpd_polygons): sys.exit()

    gs = gpd.GeoSeries(gpd_polygons)
    gs.index.name = "region"

    gs.to_file(rfile, driver = "GeoJSON")
    gs.crs = None

    return gs


regions = build_regions(args.video, args.regions)

if args.no_count: sys.exit()



## Get all tracks
tr = pd.read_csv(args.tracks, usecols = ["o", "x", "y", "t"])
pts = gpd.GeoSeries([Point(xy) for xy in zip(tr.x, tr.y)], crs = None)
tr = gpd.GeoDataFrame(data = tr, geometry = pts)


## Map these to turning regions
tr = gpd.sjoin(tr, regions, op = "within")
tr = tr.rename(columns = {"index_right" : "label"})

## Count hits in each region.
turning = tr.groupby(["o", "label"])\
             .agg(n = pd.NamedAgg(column='o', aggfunc='count'),
                  t = pd.NamedAgg(column='t', aggfunc='mean'))

## Requirement on time in each region.
turning.query(f"n >= {MIN_DET}", inplace = True)

## If there are at least two regions, the first is entrance and the latest is exit.
turning = turning.reset_index().pivot(index = "o", columns = "label", values = "t")
turning = turning[(~turning.isna()).sum(axis = 1) >= 2].copy()
turning = pd.DataFrame({"enter" : turning.idxmin(axis = 1), 
                        "exit"  : turning.idxmax(axis = 1)})


## Get turning regions
region_labels = set(regions.index)

## Possible perumutations of these.
possible_turns = itertools.permutations(region_labels, 2)

## Finally, count over the possible turning permutations.
counts = []
for I, O in possible_turns:
    N = turning.query(f"(enter == '{I}') & (exit == '{O}')").shape[0]
    counts.append((I, O, N))

## Write these to file
counts = pd.DataFrame(counts, columns = ["in", "out", "count"])
counts.to_csv("counts.csv", index = False)

## And let's see :-)
print(counts)


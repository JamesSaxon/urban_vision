#!/usr/bin/env python

import cv2 
import os, sys, re
from time import sleep
from copy import deepcopy as dc

from tqdm import tqdm 

import pandas as pd
import geopandas as gpd

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video', type = str, default = "/home/jsaxon/proj/vid/burnham/sept/20200918_065152.MOV")
parser.add_argument('--odir', type = str, default = "../data/val/")
parser.add_argument('-r', '--rate', default = 1, type = int)
parser.add_argument('-w', '--overwrite', action = "store_true", default = False)
parser.add_argument('-p', '--proceed', action = "store_true", default = False)
parser.add_argument('--regions', type = str, default = "")
args  = parser.parse_args()


video = args.video
opath = video.split("/")[-1]
opath = opath.lower().replace(".mp4", ".mov")
opath = opath.replace(".mov", ".csv")
opath = args.odir + opath

rate = args.rate
TOTAL = 5400

os.makedirs(args.odir, exist_ok = True)

if os.path.exists(opath):

    if not args.overwrite and not args.proceed:
        print(f"File {opath} already exists; specify --overwrite or --proceed.")
        sys.exit()

    if args.overwrite:
        os.remove(opath)

  
vid = cv2.VideoCapture(video)

FRAMEX = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAMEY = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

SIZEX  = 1200
SCALE  = FRAMEX / SIZEX
SIZEY  = FRAMEY / SCALE

FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_regions(rfile = args.regions):

    if not rfile: return None

    if os.path.isfile(rfile):

        regions = gpd.read_file(rfile).set_index("region")
        regions.crs = None

        poly = regions.geometry.apply(lambda x : (np.array(x.exterior.coords) / SCALE).astype(int))

        return poly

    return None

  
def print_count_on_frame(frame, entrances, exits, frame_entrances = None, frame_exits = None):

    counter_frame = dc(frame)

    counter_frame = cv2.putText(counter_frame, "{:05d}".format(exits), 
                                org = (int(0.02 * SIZEX), int(0.06 * SIZEY)),
                                fontFace = FONT, fontScale = 1, color = (255, 255, 255), thickness = 2)

    counter_frame = cv2.putText(counter_frame, "EXITS", 
                                org = (int(0.02 * SIZEX), int(0.10* SIZEY)),
                                fontFace = FONT, fontScale = 1, color = (255, 255, 255), thickness = 2)

    counter_frame = cv2.putText(counter_frame, "     {:05d}".format(entrances),
                                org = (int(0.83 * SIZEX), int(0.06 * SIZEY)),
                                fontFace = FONT, fontScale = 1, color = (255, 255, 255), thickness = 2)

    counter_frame = cv2.putText(counter_frame, "ENTRANCES",
                                org = (int(0.83 * SIZEX), int(0.10 * SIZEY)),
                                fontFace = FONT, fontScale = 1, color = (255, 255, 255), thickness = 2)

    if frame_exits is not None:

        counter_frame = cv2.putText(counter_frame, "EXITS",
                                    org = (int(0.02 * SIZEX), int(0.83 * SIZEY)),
                                    fontFace = FONT, fontScale = 1, color = (255, 0, 0), thickness = 2)

        counter_frame = cv2.putText(counter_frame, "{:05d}".format(frame_exits), 
                                    org = (int(0.02 * SIZEX), int(0.93 * SIZEY)),
                                    fontFace = FONT, fontScale = 2, color = (255, 0, 0), thickness = 3)

        counter_frame = cv2.putText(counter_frame, "ENTRANCES",
                                    org = (int(0.815 * SIZEX), int(0.83 * SIZEY)),
                                    fontFace = FONT, fontScale = 1, color = (255, 0, 0), thickness = 2)

        counter_frame = cv2.putText(counter_frame, "{:05d}".format(frame_entrances),
                                    org = (int(0.80 * SIZEX), int(0.93 * SIZEY)),
                                    fontFace = FONT, fontScale = 2, color = (255, 0, 0), thickness = 3)

    return counter_frame


regions = get_regions(args.regions)


nframe = 0

entrances, exits = 0, 0

counts = []
if os.path.exists(opath):

    df = pd.read_csv(opath)
    counts = df.values.tolist()

    nframe, frame_entrances, frame_exits = counts[-1]

    if not frame_entrances and not frame_exits:
        print("Appears to have completed file -- do not proceed.")
        sys.exit()

    entrances = df.entrances.sum()
    exits = df.exits.sum()

    print(f"fast fowarding to frame {nframe}")
    for f in tqdm(range(nframe)): vid.read()


PAUSE = True

pbar = tqdm(total = TOTAL, desc = video)
pbar.update(nframe)
while True: 

    # reading from frame 
    ret, frame = vid.read() 
    pbar.update(1)
    
    if not ret:
        counts.append([nframe, 0, 0])
        break

    if (nframe % rate) and (nframe > 100) and (nframe < TOTAL - 100): 

        nframe += 1
        continue

    frame = cv2.resize(frame, None, fx = 1 / SCALE, fy = 1 / SCALE,
                       interpolation = cv2.INTER_AREA)

    frame = cv2.polylines(frame, regions, True, (255, 255, 255), 1)

    counter_frame = print_count_on_frame(frame, entrances, exits)

    cv2.imshow('frame', counter_frame)

    key = cv2.waitKey(10)

    # Play and pause to count entrances and exits
    if key in [ord(" "), ord("i"), ord("x")] or PAUSE:

        PAUSE = False

        frame_entrances = 0
        frame_exits = 0

        if key == ord("i"): frame_entrances = 1
        if key == ord("x"): frame_exits     = 1

        while True:

            key = cv2.waitKey(10)
            if key in [ord("q"), ord(" ")]:
                break

            if key == ord("i"):
                frame_entrances += 1
            if key == ord("I"):
                frame_entrances -= 1
                if frame_entrances < 0:
                    frame_entrances = 0

            if key == ord("x"):
                frame_exits += 1
            if key == ord("X"):
                frame_exits -= 1
                if frame_exits < 0:
                    frame_exits = 0


            counter_frame = print_count_on_frame(frame, entrances + frame_entrances, exits + frame_exits, 
                                                 frame_entrances, frame_exits)

            cv2.imshow('frame', counter_frame)

        entrances += frame_entrances
        exits += frame_exits

        if frame_entrances or frame_exits:
            counts.append([nframe, frame_entrances, frame_exits])

        df = pd.DataFrame(data = counts, columns = ["frame", "entrances", "exits"])
        df.set_index("frame", inplace = True)
        print(df.tail(3))

    nframe += 1

    # Quit also... can quit _from_ pause...
    if key == ord("q"):
        break

pbar.close()

vid.release() 
cv2.destroyAllWindows() 

df = pd.DataFrame(counts, columns = ["frame", "entrances", "exits"])
df.to_csv(opath, index = False)



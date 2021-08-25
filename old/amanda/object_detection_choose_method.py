import os, sys, re, glob, cv2, numpy as np
import time

from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

import tracker
import detector

tracker = tracker.Tracker(color=(255, 0, 255), max_distance = 300)

video = "/Users/amandawhaley/Projects/UrbanVision/lsd_cars.mov"

opath = re.sub(r".*\/(.*).mov", r"\1/", video)
model = '/Users/amandawhaley/Projects/UrbanVision/coral/tflite/python/examples/detection/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
_label = '/Users/amandawhaley/Projects/UrbanVision/coral/tflite/python/examples/detection/models/coco_labels.txt'
view = True
panels = False

engine = DetectionEngine(model)
labels = dataset_utils.read_label_file(_label)
nframe = 0

HISTORY = 250
BURN_IN = 25
NFRAMES = 1000

detector = detector.Detector(model = model, labels = _label, categs = ["car", "truck"], thresh = 0.65, k = 3)
detector.set_bkd(video, history = HISTORY, burn_in = BURN_IN, nframes = NFRAMES)

thresh = 0.65

SCALE = 4
KERNEL = 60 // SCALE
if not KERNEL % 2: KERNEL +=1

def resize(img, resize = SCALE):
    return cv2.resize(img, None, fx = 1 / resize, fy = 1 / resize, interpolation = cv2.INTER_AREA)

def color(img, color = cv2.COLORMAP_PARULA):
    return cv2.applyColorMap(img, color)


vid = cv2.VideoCapture(video)
nframe = 0
t0= time.clock()
XY = None
for nframe in tqdm(range(NFRAMES), desc = "Video"):
    # reading from frame
    ret, frame = vid.read()
    if not ret:
        if view: print("Ran out of frames....")
        break
    positions, img = detector.detect_objects(frame, panels=panels, view=view)

    if len(positions): XY = positions
    tracker.update(XY)
    if view:
        img = tracker.draw(img, depth=50)
        cv2.imshow("img", resize(img))
        cv2.waitKey(10)

elapsed_time=time.clock()-t0
frame_rate=nframe/elapsed_time
print("Frame rate: ", frame_rate)
filename = "tracker_output_contour"
if panels: filename += "_panels"
filename += ".csv"
cv2.destroyAllWindows()
cv2.waitKey(10)
tracker.write(filename)

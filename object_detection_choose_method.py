import os, sys, re, glob, cv2, numpy as np
import time

from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

import tracker

tracker = tracker.Tracker()

video = "/Users/amandawhaley/Projects/UrbanVision/lsd_cars.mov"

opath = re.sub(r".*\/(.*).mov", r"\1/", video)
model = '/Users/amandawhaley/Projects/UrbanVision/coral/tflite/python/examples/detection/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
_label = '/Users/amandawhaley/Projects/UrbanVision/coral/tflite/python/examples/detection/models/coco_labels.txt'
view = True
panels = False

engine = DetectionEngine(model)
labels = dataset_utils.read_label_file(_label)
nframe = 0


vid = cv2.VideoCapture(video)
os.makedirs(opath, exist_ok=True)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

HISTORY = 250
BURN_IN = 25
NFRAMES = 1000
thresh = 0.5

# Don't burn in more than MOG stores!
BURN_IN = BURN_IN if BURN_IN < HISTORY else HISTORY
if not NFRAMES:
    while True:
        ret, frame = vid.read()
        if not ret: break
        NFRAMES += 1

    NFRAMES -= BURN_IN + 100

    vid.release()
    vid = cv2.VideoCapture(video)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
SCALE = 4
KERNEL = 60 // SCALE
if not KERNEL % 2: KERNEL +=1

mog_vid = cv2.VideoWriter(opath + 'mog.mp4', # mkv
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, # X264
                          (frame_width // SCALE, frame_height // SCALE))

def resize(img, resize = SCALE):
    return cv2.resize(img, None, fx = 1 / resize, fy = 1 / resize, interpolation = cv2.INTER_AREA)

def color(img, color = cv2.COLORMAP_PARULA):
    return cv2.applyColorMap(img, color)

bkd_mog = cv2.createBackgroundSubtractorMOG2(history = HISTORY, varThreshold = 4, detectShadows = True)
bkd_knn = cv2.createBackgroundSubtractorKNN(history = 2000)

for b in tqdm(range(BURN_IN), desc = "Burn-in"):
    ret, frame = vid.read()
    if not ret:
        print("Insufficient frames for burn-in: exiting.")
        sys.exit()

    mog_mask = bkd_mog.apply(frame)

nframe = 0
t0= time.clock()
XY = None
for nframe in tqdm(range(NFRAMES), desc = "Video"):
    # reading from frame
    ret, frame = vid.read()
    if not ret:
        if view: print("Ran out of frames....")
        break
    positions, img = tracker.detect_objects(frame, bkd_mog, engine, panels=panels, view=view)

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
tracker.write_out(filename)

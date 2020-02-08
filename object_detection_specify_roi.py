import argparse
import platform
import subprocess
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from glob import glob
import sys, tqdm
import pandas as pd

colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]

model = '/Users/amandawhaley/Projects/UrbanVision/coral/tflite/python/examples/detection/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
_label = '/Users/amandawhaley/Projects/UrbanVision/coral/tflite/python/examples/detection/models/coco_labels.txt'
inp = '/Users/amandawhaley/Projects/UrbanVision/lsd_cars.mov'
_roi = True
scale = 4
no_output = False
frames = 0
thresh = 0.5
keep_aspect_ratio = False
k = 10
verbose = True
categs = []
view = True

engine = DetectionEngine(model)
labels = dataset_utils.read_label_file(_label) if _label else None
vid = cv2.VideoCapture(inp)
if _roi:
    ret, img = vid.read()
    vid.release()
    if not ret:
        print("Video file not valid")
        sys.exit()

    if scale != 1:
        img = cv2.resize(img, None, fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_AREA)

    ROI = cv2.selectROI(img)
    cv2.destroyWindow("ROI selector")

    ROI = [int(x * scale) for x in ROI]

    XMIN, XMAX = ROI[0], ROI[0] + ROI[2]
    YMIN, YMAX = ROI[1], ROI[1] + ROI[3]

    #Set ROI manually to compare with contours method
    XMIN, XMAX = 1200, 2699
    YMIN, YMAX = 300, 1199

    vid = cv2.VideoCapture(inp)

if _roi:
    shade = 2 * np.ones((int(vid.get(4) / scale), int(vid.get(3) / scale))).astype("uint8")
    shade[int(YMIN/scale):int(YMAX/scale),int(XMIN/scale):int(XMAX/scale)] -= 1

nframe = 0
detected = []

while True:
    ret, frame = vid.read()
    nframe += 1
    print(nframe, end = " ", flush = True)
    if not ret: break
    if frames and nframe > frames: break
    roi = frame if not _roi else frame[YMIN:YMAX, XMIN:XMAX]
    scaled = cv2.resize(frame, None, fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_AREA)
    if _roi: scaled = (scaled / shade[:,:,np.newaxis]).astype("uint8")
    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # Run inference.
    ans = engine.detect_with_image(img, threshold = thresh, keep_aspect_ratio=keep_aspect_ratio, relative_coord=False, top_k = k)

    # Save result.
    if verbose: print('=========================================')
    if ans:
        for obj in ans:
            label = None
            if labels is not None and len(categs):
                label = labels[obj.label_id]
                if label not in categs: continue

            if label is not None and verbose:
                print(labels[obj.label_id] + ",", end = " ")

            # Draw a rectangle.
            box = obj.bounding_box.flatten()
            color = colors[categs.index(label)] if len(categs) else (0, 0, 255)

            if _roi:
                box[0] += XMIN
                box[2] += XMIN
                box[1] += YMIN
                box[3] += YMIN
            draw_box = (box / scale).astype(int)
            cv2.rectangle(scaled, tuple(draw_box[:2]), tuple(draw_box[2:]), color, 2)
            if verbose:
                print('conf. = ', obj.score)
                print('-----------------------------------------')
            detected.append([nframe, label, obj.score, box[0], box[1], box[2], box[3]])

    else:
        if verbose: print('No object detected!')

    if view:
        cv2.imshow("view", scaled)
        if (cv2.waitKey(1) & 0xff) == 27: break

    if out is not None: out.write(scaled)
    if verbose: print('Recorded frame {}'.format(nframe))
cv2.destroyAllWindows()

print(len(detected), " objects detected")

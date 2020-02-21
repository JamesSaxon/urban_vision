import os, sys, re, glob, cv2, numpy as np
import time

from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

import tracker
import detector



video = "/Users/amandawhaley/Projects/UrbanVision/lsd_cars.mov"

opath = re.sub(r".*\/(.*).mov", r"\1/", video)
model = '/Users/amandawhaley/Projects/UrbanVision/coral/tflite/python/examples/detection/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
_label = '/Users/amandawhaley/Projects/UrbanVision/coral/tflite/python/examples/detection/models/coco_labels.txt'
view = False
panels = False
scale = 4
gauss = True
thresh = 0.65

scales_to_test = [1,2,3,4,5,6,7,8,9,10]
gauss_to_test = [True, False]
panels_to_test = [True, False]
thresh_to_test = [0.5, 0.6, 0.7]

data = []

labels = dataset_utils.read_label_file(_label)

HISTORY = 250
BURN_IN = 25
NFRAMES = 1000

for scale in scales_to_test:
    for gauss in gauss_to_test:
        for panels in panels_to_test:
            for thresh in thresh_to_test:
                tracker_test = tracker.Tracker(color=(255, 0, 255), max_distance = 300)
                detector_test = detector.Detector(model = model, labels = _label, categs = ["car", "truck"], thresh = thresh, k = 3)
                detector_test.set_bkd(video, history = HISTORY, burn_in = BURN_IN, nframes = NFRAMES)
                KERNEL = 60 // scale
                if not KERNEL % 2: KERNEL +=1

                def resize(img, resize = scale):
                    return cv2.resize(img, None, fx = 1 / resize, fy = 1 / resize, interpolation = cv2.INTER_AREA)

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
                    positions, img = detector_test.detect_objects(frame, panels=panels, view=view, scale=scale)

                    if len(positions): XY = positions
                    tracker_test.update(XY)
                    if view:
                        img = tracker_test.draw(img, depth=50)
                        cv2.imshow("img", resize(img))
                        cv2.waitKey(10)

                cv2.destroyAllWindows()
                cv2.waitKey(10)
                elapsed_time=time.clock()-t0
                frame_rate=nframe/elapsed_time
                filename = "sc_" + str(scale) + "_g_" + str(gauss)[0] + "_p_" + str(panels)[0] + "_t_" + str(thresh) + ".csv"
                data.append([scale, gauss, panels, thresh, frame_rate, filename])
                tracker_test.write(filename)
                print(data[-1])

data_out = pd.concat([test.df for test in data])

data_out.to_csv('object_detection_test_data.csv', index = False)

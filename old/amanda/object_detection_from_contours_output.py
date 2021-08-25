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
view = False

engine = DetectionEngine(model)
labels = dataset_utils.read_label_file(_label)
nframe = 0
detected = {}
test_sample = {}

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
gray, last, last_gray = None, None, None
for nframe in tqdm(range(NFRAMES), desc = "Video"):
    # reading from frame
    ret, frame = vid.read()
    if not ret:
        if view: print("Ran out of frames....")
        break
    scaled_frame = resize(frame)
    mog = bkd_mog.apply(scaled_frame)

    if last is None:
        last = scaled_frame
        gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        continue

    mog[mog < 129] = 0
    mog = cv2.GaussianBlur(mog, (KERNEL, KERNEL), 0)
    mog[mog < 50] = 0
    mog_color = color(mog)

    # find contours in the binary image
    mog_mask = ((mog > 128) * 255).astype("uint8")
    img = frame

    _, contours, _ = cv2.findContours(mog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas_inferenced = set()
    positions = []
    XY = None
    for c in contours:
        if cv2.contourArea(c) < 75: continue

        #Calculate bounding boxes for each countour
        x, y, w, h = cv2.boundingRect(c)
        width = max(300,w*2*SCALE)
        height = max(300,h*2*SCALE)
        x_mid = (x + w//2)*SCALE
        y_mid = (y + h//2)*SCALE

        xmin = max(0,x_mid - width//2)
        xmax = x_mid + width//2
        ymin = max(0,y_mid - height//2)
        ymax = y_mid + height//2
        #print(xmin, xmax, ymin, ymax)

        #Run inference
        roi = frame[ymin:ymax, xmin:xmax]
        image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        ans = engine.detect_with_image(image, threshold = thresh, keep_aspect_ratio=False, relative_coord=False, top_k = 10)

        #Draw rectangle around roi
        #cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,255,0), 3)

        # Save result.
        if view: print('=========================================')
        if ans:
            for obj in ans:
                label = labels[obj.label_id]
                #print(labels[obj.label_id] + ",", end = " ")

                # Draw a rectangle.
                box = obj.bounding_box.flatten()
                box[0] += xmin
                box[2] += xmin
                box[1] += ymin
                box[3] += ymin
                draw_box = box.astype(int)

                if view:
                    cv2.rectangle(img, tuple(draw_box[:2]), tuple(draw_box[2:]), (0,255,255), 2)
                    print('conf. = ', obj.score)
                    print('-----------------------------------------')
                detected[nframe] = [label, obj.score, box[0], box[1], box[2], box[3]]
                positions.append((int((box[0] + box[2])//2), int((box[1] + box[3])//2)))

    if len(positions): XY = positions
    tracker.update(XY)
    if view:
        img = tracker.draw(img, depth=50)
        cv2.imshow("img", resize(img))
        cv2.waitKey(10)

elapsed_time=time.clock()-t0
frame_rate=nframe/elapsed_time
print("Frame rate: ", frame_rate)
cv2.destroyAllWindows()
cv2.waitKey(10)
tracker.write_out('tracker_output_contour.csv')

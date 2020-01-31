#!/Users/jsaxon/anaconda/envs/cv/bin/python

import os, sys, re, glob, cv2, numpy as np

from skimage.measure import compare_ssim

from tqdm import tqdm

video = "/Users/jsaxon/Documents/Chicago/Chalkboard/akaso/VIDEO/53_hp_morning.mov"

if len(sys.argv) > 1:
  video = sys.argv[1]
  print(video)

opath = re.sub(r".*\/(.*).mov", r"\1/", video)
  
vid = cv2.VideoCapture(video)
os.makedirs(opath, exist_ok=True) 

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

HISTORY = 100 
BURN_IN = 100
NFRAMES = 10000

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

avg_vid = cv2.VideoWriter(opath + 'avg.mp4', # mkv
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, # X264
                          (frame_width // SCALE, frame_height // SCALE))

mog_vid = cv2.VideoWriter(opath + 'mog.mp4', # mkv
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, # X264
                          (frame_width // SCALE, frame_height // SCALE))

bkd_vid = cv2.VideoWriter(opath + 'bkd.mp4', # mkv
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, # X264
                          (frame_width // SCALE, frame_height // SCALE))

last_vid = cv2.VideoWriter(opath + 'last.mp4', # mkv
                           cv2.VideoWriter_fourcc(*'mp4v'), 30, # X264
                           (frame_width // SCALE, frame_height // SCALE))



def resize(img, resize = SCALE):

    return cv2.resize(img, None, fx = 1 / resize, fy = 1 / resize, interpolation = cv2.INTER_AREA)


def color(img, color = cv2.COLORMAP_PARULA):

    return cv2.applyColorMap(img, color)



bkd_mog = cv2.createBackgroundSubtractorMOG2(history = HISTORY, varThreshold = 16, detectShadows = True)

for b in tqdm(range(BURN_IN), desc = "Burn-in"):

    ret, frame = vid.read() 
    if not ret:
        print("Insufficient frames for burn-in: exiting.")
        sys.exit()

    # frame = resize(frame)
    mog_mask = bkd_mog.apply(resize(frame))

nframe = 0
gray, last, last_gray = None, None, None
for nframe in tqdm(range(NFRAMES), desc = "Video"):
      
    # reading from frame 
    ret, frame = vid.read() 

    if not ret: 

        print("Ran out of frames....")
        break

    frame = resize(frame)
    mog_mask = bkd_mog.apply(frame)

    if last is None:
        last = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        continue

    mog_mask[mog_mask < 129] = 0

    mog_mask = cv2.GaussianBlur(mog_mask, (KERNEL, KERNEL), 0)
    mog_mask[mog_mask < 50] = 0

    mog_vid.write(color(mog_mask))
    continue

    last, last_gray = frame, gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    mog_bkd = bkd_mog.getBackgroundImage()
    mog_gray = cv2.cvtColor(mog_bkd, cv2.COLOR_BGR2GRAY)
    
    _, bkd_ssim  = compare_ssim(gray, mog_gray, full = True)
    bkd_ssim = ((bkd_ssim < 0.8) * 255).astype("uint8") 
    bkd_ssim = cv2.erode(bkd_ssim, None, iterations = 1)
        
    _, last_ssim = compare_ssim(gray, last_gray, full = True)
    last_ssim = ((last_ssim < 0.4) * 255).astype("uint8") 
    last_ssim = cv2.erode(last_ssim, None, iterations = 2)
    
    ##  diff_last = np.abs(gray - last_gray)
    ##  diff_last = cv2.erode(diff_last, None, iterations = 2)
    ##  diff_last = cv2.GaussianBlur(diff_last, (25, 25), 0)

    avg = (mog_mask.astype(int) + bkd_ssim + last_ssim)
    avg = avg * 256 / avg.max()
    avg = avg.astype("uint8")


    avg_vid .write(color(avg))
    mog_vid .write(color(mog_mask))
    bkd_vid .write(color(bkd_ssim))
    last_vid.write(color(last_ssim))



avg_vid .release()
mog_vid .release()
bkd_vid .release()
last_vid.release()
  
vid.release() 
cv2.destroyAllWindows() 


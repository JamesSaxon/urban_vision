#!/home/jsaxon/anaconda3/envs/cv/bin/python

import cv2, sys, pandas as pd

vid_dir = "../../../data/cv/vid/"
stream = cv2.VideoCapture(vid_dir + "lsd_e_20200131.mov")

geo_dir = "../../../data/cv/geo/"
locations = pd.read_csv(geo_dir + "lsd_local.csv")
locations["xp"] = None
locations["yp"] = None

ret, img = stream.read()
stream.release()

if not ret: sys.exit()

SCALE = 2.8
img = cv2.resize(img, None, fx = 1 / SCALE, fy = 1 / SCALE, interpolation = cv2.INTER_AREA)

coords = []
def click_and_id(event, x, y, flags, param):
    
    if event is not cv2.EVENT_LBUTTONDOWN: return

    locations.loc[ri, "xp"] = round(x * SCALE, 2)
    locations.loc[ri, "yp"] = round(y * SCALE, 2)

    draw_points_on_image()

def draw_points_on_image():

    img_pts = img.copy()
    for _, row in locations[:ri+1].iterrows():

        if row.xp is None or row.yp is None: continue

        coords = (int(row.xp / SCALE), int(row.yp / SCALE))
        cv2.circle(img_pts, coords, 5, (0, 0, 255), 2)
        cv2.putText(img_pts, row.id, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("click", img_pts)


cv2.imshow("click", img)
cv2.setMouseCallback("click", click_and_id)

ri, last, rmax = 0, None, locations.shape[0]
while ri < rmax:

    key = cv2.waitKey(30) & 0xff

    if key == 27: break
    if key == ord("q"): break

    if key == ord("s"): 

        ri += 1
        continue

    if key == ord("u"):

        if ri: ri -= 1

        locations.loc[ri, "xp"] = None
        locations.loc[ri, "yp"] = None
        draw_points_on_image()


    if locations.loc[ri, "xp"] is not None:

        ri += 1

    if last != ri:
        print(locations.loc[ri, "id"])
        last = ri


locations.dropna().to_csv(geo_dir + "lsd_e_20200131.csv", index = False)


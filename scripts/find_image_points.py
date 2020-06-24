#!/Users/jsaxon/anaconda/envs/cv/bin/python

import cv2, sys, pandas as pd

def click_and_id(event, x, y, flags, param):

    global locations
    
    if event is not cv2.EVENT_LBUTTONDOWN: return

    scale = param[0]
    locations.loc[ri, "xp"] = round(x * scale, 2)
    locations.loc[ri, "yp"] = round(y * scale, 2)

    draw_points_on_image(scale)

def draw_points_on_image(scale):

    img_pts = img.copy()
    for _, row in locations[:ri+1].iterrows():

        if row.xp is None or row.yp is None: continue

        coords = (int(row.xp / scale), int(row.yp / scale))
        cv2.circle(img_pts, coords, 5, (0, 0, 255), 2)
        cv2.putText(img_pts, row.id, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("click", img_pts)


locations, ri, img = None, None, None
def main(geog, video, skip = 0, scale = 1):

    global locations, ri, img

    stream = cv2.VideoCapture(video)
    
    locations = pd.read_csv(geog)

    locations["xp"] = None
    locations["yp"] = None
    
    for s in range(skip+1): ret, img = stream.read()
    stream.release()
    
    if not ret:
        print("Failed to read video stream -- exiting.")
        sys.exit()
    
    img = cv2.resize(img, None, fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_AREA)
    
    coords = []
    
    cv2.imshow("click", img)

    cv2.setMouseCallback("click", click_and_id, [scale])
    
    ri, last, rmax = 0, None, locations.shape[0]
    while ri < rmax:

        if locations.loc[ri, "id"] == "CAM":
          ri += 1
          continue
    
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
            draw_points_on_image(scale)
    
    
        if locations.loc[ri, "xp"] is not None:
    
            ri += 1
    
        if last != ri and ri < rmax:
            print(locations.loc[ri, "id"])
            last = ri

    locations["xp"] = locations["xp"].astype(int)
    locations["yp"] = locations["yp"].astype(int)
    print(locations)
    
    locations.to_csv(geog, float_format = "%.6f", index = False)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Label points in a video.')
    parser.add_argument('--geog', type = str, required = True)
    parser.add_argument("--video", type = str, required = True)
    parser.add_argument("--scale", type = float, default = 1.0)
    parser.add_argument("--skip", type = int, default = 0)
    args = parser.parse_args()

    main(**vars(args))





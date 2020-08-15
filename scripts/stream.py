#!/usr/bin/env python3

import configargparse
import platform
import subprocess

import cv2

import numpy as np

from PIL import Image
from PIL import ImageDraw

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from glob import glob

import sys

from tqdm import tqdm

import pandas as pd

import tracker as tr, detector as det

import threading
import queue

colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]


def get_roi(vinput, scale, roi, select_roi = False): 

    vid = cv2.VideoCapture(vinput)

    FRAMEX, FRAMEY = int(vid.get(3)), int(vid.get(4))

    ROI = []
    if select_roi or len(roi):

        ret, img = vid.read()

        if not ret:
            print("Video file '{}' is not valid.".format(vinput))
            sys.exit()

        if select_roi:

            if scale != 1:
                img = cv2.resize(img, None, fx = 1 / scale, fy = 1 / scale)

            ROI = cv2.selectROI(img)
            cv2.destroyWindow("ROI selector")

            ROI = [int(x * scale) for x in ROI]

            XMIN, XMAX = ROI[0], ROI[0] + ROI[2]
            YMIN, YMAX = ROI[1], ROI[1] + ROI[3]

            print(XMIN / FRAMEX, XMAX / FRAMEX, YMIN / FRAMEY, YMAX / FRAMEY)

        else:

            XMIN = int(FRAMEX * roi[0])
            XMAX = int(FRAMEX * roi[1])
            YMIN = int(FRAMEY * roi[2])
            YMAX = int(FRAMEY * roi[3])

        ROI = [XMIN, XMAX, YMIN, YMAX]

    else: 

        ROI = [0, FRAMEX, 0, FRAMEY]
        vid.release()

    return ROI

global COMPLETE
COMPLETE = False
def write_frame_to_stream(q, video_out):

    while not (COMPLETE and q.empty()):

        try: frame = q.get(timeout = 1)
        except: continue

        video_out.write(frame)

    q.task_done()

    video_out.release()


def main(vinput, output, 
         nframes, nskip,
         model, labels, thresh, categs, max_det_items,
         roi, xgrid, ygrid, roi_loc, max_overlap, min_area, edge_veto, roi_buffer,
         no_tracker, match_method, max_missing, max_distance, min_distance_or,
         predict_matches, kalman_track, max_track, candidate_obs,
         kalman_viz, contrail, scale, view, no_output, no_video_output, pretty_video, ofps,
         show_heat, heat_frac, heat_fade, geometry, verbose, 
         select_roi, config):


    ##  Open the input video file...
    vid = cv2.VideoCapture(vinput)
    
    FRAMEX, FRAMEY = int(vid.get(3)), int(vid.get(4))

    ##  Shape parameters.
    ROI = roi
    XMIN,  XMAX,  YMIN,  YMAX  = ROI
    XMINS, XMAXS, YMINS, YMAXS = [int(v/scale) for v in ROI]


    ##  And the output video file.
    video_out = None
    if not no_video_output:

        video_out = cv2.VideoWriter(output + "_det.mp4", cv2.VideoWriter_fourcc(*'mp4v'), ofps,
                                    (round(FRAMEX / scale), round(FRAMEY / scale)))

        qproc = queue.Queue(20)
        output_thread = threading.Thread(target = write_frame_to_stream, args = (qproc, video_out))
        output_thread.start()

        if pretty_video:

            shade = 1.5 * np.ones((round(FRAMEX / scale), round(FRAMEY / scale))).astype("uint8")
            shade[YMINS:YMAXS,XMINS:XMAXS] = 1


    ##  Build the detector, including the TF engine.
    detector = det.Detector(model = model, labels = labels, categs = categs, thresh = thresh, k = max_det_items,
                            max_overlap = max_overlap, loc = roi_loc, edge_veto = edge_veto, min_area = min_area, verbose = False)

    if geometry and show_heat: detector.set_world_geometry(geometry)

    ##  And finally, the tracker.
    tracker = None
    if not no_tracker:
        tracker = tr.Tracker(method = match_method, roi_loc = roi_loc,
                             max_missing = max_missing, candidate_obs = candidate_obs, max_track = max_track, 
                             max_distance = max_distance, min_distance_overlap = min_distance_or,
                             predict_match_locations = predict_matches, kalman_track_cov = kalman_track, 
                             kalman_viz_cov = kalman_viz, contrail = contrail)

        tracker.set_roi({"xmin" : XMIN, "xmax" : XMAX, "ymin" : YMIN, "ymax" : YMAX},
                        roi_buffer = (YMAX - YMIN) * roi_buffer)


    nframe = 0
    pbar = tqdm(total = nframes)
    while True:

        ret, frame = vid.read()

        if not ret or nframe >= nframes:
            break

        if nframe < nskip:
            nframe += 1
            continue

        detections = detector.detect_grid(frame, ROI, xgrid = xgrid, ygrid = ygrid)
        if tracker: tracker.update(detections, frame)

        if view or video_out:

            scaled = cv2.resize(frame, None, fx = 1 / scale, fy = 1 / scale)

            if show_heat: 

                if not nframe % 10:
                    mask, heat = detector.heatmap(size = scaled.shape[:2], scale = scale, xmin = 50)

                if nframe >= heat_fade: heat_frac = heat_frac
                else: heat_frac = heat_frac * nframe / heat_fade

                if not ROI:
                    scaled[mask] = (scaled[mask] * (1 - heat_frac) + heat[mask] * heat_frac).astype("uint8")

                else:
                    scaled[YMINS:YMAXS,XMINS:XMAXS][mask[YMINS:YMAXS,XMINS:XMAXS]] = \
                        (scaled[YMINS:YMAXS,XMINS:XMAXS][mask[YMINS:YMAXS,XMINS:XMAXS]] * (1 - heat_frac) + \
                         heat  [YMINS:YMAXS,XMINS:XMAXS][mask[YMINS:YMAXS,XMINS:XMAXS]] * heat_frac).astype("uint8")

            ##  Visualize.  If tracker is running, use it; otherwise let the detector do it.
            if tracker: scaled = tracker .draw(scaled, scale = scale)
            else:       scaled = detector.draw(scaled, scale = scale, width = 1, color = (255, 255, 255))

            ##  Cut out the detection region.
            if ROI:

                if pretty_video: scaled = (scaled / shade[:,:,np.newaxis]).astype("uint8")
                else: scaled = cv2.rectangle(scaled, tuple((XMINS, YMINS)), tuple((XMAXS, YMAXS)), (0, 0, 0), 3)


            ##  Write it, if the stream has not been turned off.
            scaled = cv2.putText(scaled, "{:05d}".format(nframe), org = (int(FRAMEX*0.02 / scale), int(FRAMEY*0.98 / scale)),
                                 fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2 if FRAMEX / scale > 600 else 1,
                                 color = (255, 255, 255), thickness = 2)

            ##  View it, if relevant.
            if view:

                cv2.imshow("view", scaled)
                if (cv2.waitKey(1) & 0xff) == 27: break

            ##  Otherwise write it out.
            ## if video_out: video_out.write(scaled)
            qproc.put((scaled))

        nframe += 1
        pbar.update()

    pbar.close()


    ## Finalize all outputs -- video stream, detector, and tracker.
    if not no_video_output: 

        global COMPLETE
        COMPLETE = True

        output_thread.join()
        video_out.release()

    if not no_output:

        ## detector.write(output + "_det.csv")

        if tracker: tracker.write(output + "_tr.csv")




if __name__ == '__main__':

    parser = configargparse.ArgParser(default_config_files = ['./stream_defaults.conf'])
    parser.add('-c', '--config', required = False, is_config_file = True, help = 'Path for config file.')

    ## Basic inputs
    parser.add('-i', '--vinput', help = 'Directory of input images.', required=True)
    parser.add('-o', '--output', help = 'Name of output file, by default derived from input name.', default = "")

    ## What to process
    parser.add("-f", "--nframes", default = float("inf"), type = float, help = "Total number of frames to process")
    parser.add("--nskip", default = 0, type = int, help = "Skip deeper into a stream (useful for debugging)")

    ## Model / Detection Engine
    parser.add('--model', required = True, help = "The tflite model")
    parser.add('--labels', required = True, help='Path of the labels file.')
    parser.add('--thresh', default = 0.4, type = float, help = "Confidence level of detections")
    parser.add("--categs", default = ["car"], nargs = "+", help = "Types of objects to process -- must match labels")
    parser.add('--max_det_items', default = 50, type = int, help = "'k' parameter of max detections, for engine.")

    ## Detector Parameters
    parser.add("--select_roi", default = False, action = "store_true", help = "Interactively re-select the ROI.")
    parser.add("--roi", default = [], type = float, nargs = 4, help = "xmin, xmax, ymin, ymax")
    parser.add("--xgrid", default = 1, type = int, help = "Number of times to subdivide the detection ROI horizontally.")
    parser.add("--ygrid", default = 1, type = int, help = "Number of times to subdivide the detection ROI vertically.")
    parser.add("--roi_loc", default = "upper center", type = str, help = "Location on a detection bounding box to use.")
    parser.add("--max_overlap", default = 0.25, type = float, help = "Maximum that detections can overlap, as max(A_i / Intersection).")
    parser.add("--min_area", default = 0, type = float, help = "Minimum size of a detection.")
    parser.add("--edge_veto", default = 0.005, type = float, help = "Veto detections within this fractional distance of the edge of a (sub) ROI when DETECTING.")
    parser.add("--roi_buffer", default = 0.02, type = float, help = "Ignore new objects or delete existing ones, within this fractional distance of the ROI, when TRACKING.")

    ## Tracker Parameters
    parser.add("--no_tracker", default = False, action = "store_true", help = "Whether to turn off the tracker.")
    parser.add("--match_method", default = "min_cost", type = str, choices = ["min_cost", "greedy"], help = "Method for associating objects between frames.")
    parser.add("--max_missing", default = 5, type = int, help = "Number of frames that an object can go missing, before being deleted.")
    parser.add("--max_distance", default = 1.0, type = float, help = "Maximum distance that an object can move as a proportion of sqrt(A)")
    parser.add("--min_distance_or", default = 0.4, type = float, help = "Minimum distance between detections, in units of sqrt(A).")
    parser.add("--predict_matches", default = False, action = "store_true", help = "Do or do not use Kalman filtering to predict new match locations.")
    parser.add("--kalman_track", default = 0, type = int, help = "Scale of Kalman error covariance in pixels for feed-forward tracking.")
    parser.add("--max_track", default = 0, type = int, help = "How many frames to track an object, using correlational tracking.")
    parser.add("--candidate_obs", default = 5, type = float, help = "Number of frames to privilege earlier objects.")

    ## Visualization
    parser.add("--kalman_viz", default = 0, type = int, help = "Scale of Kalman error covariance in pixels for vizualization.")
    parser.add("--contrail", default = 25, type = int, help = "Number of frames to view objects' past locations")
    parser.add("--scale", default = 1, type = float, help = "Factor by which to reduce the output size.")
    parser.add("--view", default = False, action = "store_true", help = "View the feed 'live' (or not)")
    parser.add("--no_output", default = False, action = "store_true", help = "Do not record ANY outputs -- csv or mp4.")
    parser.add("--no_video_output", default = False, action = "store_true", help = "Do not record the stream.")
    parser.add("--pretty_video", default = False, action = "store_true", help = "Shade out the un-detected space.")
    parser.add("--ofps", default = 30, type = int, help = "Output frames per second.")

    parser.add("--show_heat", action = "store_true", default = False, help = "Whether to show a projected geometry heat map.")
    parser.add("--heat_frac", type = float, default = 0.5, help = "Max alpha value")
    parser.add("--heat_fade", type = int, default = 0, help = "Number of frames, to 'fade in' geometry")
    parser.add("--geometry", type = str, default = "", help = "Filename of geometry, to calculate projection matrix.")

    parser.add("-v", "--verbose", default = False, action = "store_true", help = "")

    args = parser.parse_args()

    if args.verbose: print(parser.format_values()) ## Where did the settings come from!?

    if args.no_output: args.no_video_output = True

    if not args.output: # if it's not defined, get it from the input file.
        args.output = args.vinput.replace(".gif", "").replace(".mov", "").replace(".mp4", "")

    args.roi = get_roi(args.vinput, args.scale, args.roi, args.select_roi)

    main(**vars(args))




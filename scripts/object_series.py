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

import pandas as pd

import tracker as tr, detector as det

colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]


def main():

    parser = configargparse.ArgParser(default_config_files = ['./stream_defaults.conf'])
    parser.add('-c', '--config', required = False, is_config_file = True, help = 'Path for config file.')

    ## Basic inputs
    parser.add('-i', '--input', help = 'Directory of input images.', required=True)
    parser.add('-o', '--output', help = 'Name of output file, by default derived from input name.', default = "")

    ## What to process
    parser.add("-f", "--frames", default = 0, type = int, help = "Total number of frames to process")
    parser.add("--skip", default = 0, type = int, help = "Skip deeper into a stream (useful for debugging)")

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
    parser.add("--max_missing", default = 5, type = int, help = "Number of frames that an object can go missing, before being deleted.")
    parser.add("--max_distance", default = 1.0, type = float, help = "Maximum distance that an object can move as a proportion of sqrt(A)")
    parser.add("--min_distance_or", default = 0.4, type = float, help = "Minimum distance between detections, in units of sqrt(A).")
    parser.add("--no_tracker", default = False, action = "store_true", help = "Whether to turn off the tracker.")
    parser.add("--predict_matches", default = False, action = "store_true", help = "Do or do not use Kalman filtering to predict new match locations.")
    parser.add("--max_track", default = 0, type = int, help = "How many frames to track an object, using correlational tracking.")
    parser.add("--candidate_obs", default = 5, type = float, help = "Number of frames to privilege earlier objects.")

    ## Visualization
    parser.add("--kalman", default = 50, type = int, help = "Scale of Kalman error covariance in pixels.")
    parser.add("--contrail", default = 25, type = int, help = "Number of frames to view objects' past locations")
    parser.add("--scale", default = 1, type = float, help = "Factor by which to reduce the output size.")
    parser.add("--view", default = False, action = "store_true", help = "View the feed 'live' (or not)")
    parser.add("--no_output", default = False, action = "store_true", help = "Do not record the stream.")
    parser.add("--ofps", default = 30, type = int, help = "Output frames per second.")

    parser.add("--show_heat", action = "store_true", default = False, help = "Whether to show a projected geometry heat map.")
    parser.add("--heat_frac", type = float, default = 0.5, help = "Max alpha value")
    parser.add("--heat_fade", type = int, default = 0, help = "Number of frames, to 'fade in' geometry")
    parser.add("--geometry", type = str, default = "", help = "Filename of geometry, to calculate projection matrix.")

    parser.add("--verbose", default = False, action = "store_true", help = "")

    args = parser.parse_args()

    print(args)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())    # useful for logging where different settings came from


    # Initialize engine.
    vid = cv2.VideoCapture(args.input)

    FRAMEX, FRAMEY = int(vid.get(3)), int(vid.get(4))

    detector = det.Detector(model = args.model, labels = args.labels,
                            categs = args.categs, thresh = args.thresh, k = args.max_det_items,
                            max_overlap = args.max_overlap,
                            loc = args.roi_loc, edge_veto = args.edge_veto,
                            min_area = args.min_area,
                            verbose = False)

    if args.geometry and args.show_heat:
        detector.set_world_geometry(args.geometry)

    tracker = None
    if not args.no_tracker:
        tracker = tr.Tracker(max_missing = args.max_missing,
                             max_distance = args.max_distance,
                             min_distance_overlap= args.min_distance_or,
                             max_track = args.max_track,
                             candidate_obs = args.candidate_obs,
                             predict_match_locations = args.predict_matches,
                             kalman_cov = args.kalman,
                             contrail = args.contrail, roi_loc = args.roi_loc)

    vid = cv2.VideoCapture(args.input)


    if not args.no_output:

        output = args.input.replace("gif", "mov").replace("mov", "mp4").replace(".mp4", "_det.mp4")

        ## But if it's explicit, use that.
        if args.output: output = args.output

        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), args.ofps,
                              (round(FRAMEX / args.scale), round(FRAMEY / args.scale)))
    else: out = None


    ROI = []
    if args.select_roi or len(args.roi):

        ret, img = vid.read()
        vid.release()

        if not ret:
            print("Video file not valid")
            sys.exit()

        print("frame size", img.shape[1], img.shape[0])

        if args.select_roi:

            if args.scale != 1:
                scaled = cv2.resize(img, None,
                                    fx = 1 / args.scale, fy = 1 / args.scale,
                                    interpolation = cv2.INTER_AREA)
            else:
                scaled = img

            ROI = cv2.selectROI(scaled)
            cv2.destroyWindow("ROI selector")

            ROI = [int(x * args.scale) for x in ROI]

            XMIN, XMAX = ROI[0], ROI[0] + ROI[2]
            YMIN, YMAX = ROI[1], ROI[1] + ROI[3]

            print(XMIN / img.shape[1],
                  XMAX / img.shape[1],
                  YMIN / img.shape[0],
                  YMAX / img.shape[0])

        else:

            XMIN = int(img.shape[1] * args.roi[0])
            XMAX = int(img.shape[1] * args.roi[1])
            YMIN = int(img.shape[0] * args.roi[2])
            YMAX = int(img.shape[0] * args.roi[3])

        ROI = [XMIN, XMAX, YMIN, YMAX]

        shade = 1.5 * np.ones((int(img.shape[0] / args.scale), int(img.shape[1] / args.scale))).astype("uint8")
        shade[int(YMIN/args.scale):int(YMAX/args.scale),int(XMIN/args.scale):int(XMAX/args.scale)] = 1


        # Re-set...
        vid = cv2.VideoCapture(args.input)

    if tracker:
        if ROI:
            tracker.set_roi({"xmin" : XMIN, "xmax" : XMAX, "ymin" : YMIN, "ymax" : YMAX},
                            roi_buffer = (YMAX - YMIN) * args.roi_buffer)
        else:
            tracker.set_roi({"xmin" : 0, "xmax" : FRAMEX, "ymin" : 0, "ymax" : FRAMEY},
                            roi_buffer = FRAMEY * args.roi_buffer)

    nframe = 0
    while True:

        ret, frame = vid.read()

        print(nframe, end = " ", flush = True)

        if not ret: break
        if args.frames and nframe > args.frames: break
        if nframe < args.skip:
            nframe += 1
            continue

        detections = detector.detect_grid(frame, ROI, xgrid = args.xgrid, ygrid = args.ygrid)

        scaled = cv2.resize(frame, None,
                            fx = 1 / args.scale, fy = 1 / args.scale,
                            interpolation = cv2.INTER_AREA)

        if tracker:

            tracker.update(detections)

            if args.max_track:
                tracker.track(frame)
                tracker.reset_track(frame)

        if ROI: scaled = (scaled / shade[:,:,np.newaxis]).astype("uint8")

        if args.show_heat: 

            if not nframe % 10:
                mask, heat= detector.heatmap(size = scaled.shape[:2], scale = args.scale, xmin = 50)

            if nframe >= args.heat_fade: heat_frac = args.heat_frac
            else: heat_frac = args.heat_frac * nframe / args.heat_fade

            if not ROI:
                scaled[mask] = (scaled[mask] * (1 - heat_frac) + heat[mask] * heat_frac).astype("uint8")
            else:

                YMINS, YMAXS = int(YMIN/args.scale), int(YMAX/args.scale)
                XMINS, XMAXS = int(XMIN/args.scale), int(XMAX/args.scale)

                scaled[YMINS:YMAXS,XMINS:XMAXS][mask[YMINS:YMAXS,XMINS:XMAXS]] = \
                    (scaled[YMINS:YMAXS,XMINS:XMAXS][mask[YMINS:YMAXS,XMINS:XMAXS]] * (1 - heat_frac) + \
                     heat[YMINS:YMAXS,XMINS:XMAXS][mask[YMINS:YMAXS,XMINS:XMAXS]] * heat_frac).astype("uint8")


        ##  Visualize.  If tracker is running, use it; otherwise let the detector.
        if tracker: tracker.draw(scaled, scale = args.scale)
        else:       detector.draw(scaled, scale = args.scale, width = 1, color = (255, 255, 255))


        ##  View it, if relevant.
        if args.view:

            cv2.imshow("view", scaled)
            if (cv2.waitKey(1) & 0xff) == 27: break

        ##  Write it, if the stream has not been turned off.
        if not args.no_output:

            scaled = cv2.putText(scaled, "{:05d}".format(nframe), org = (int(FRAMEX*0.02 / args.scale), int(FRAMEY*0.98 / args.scale)),
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2 if FRAMEX / args.scale > 600 else 1,
                                color = (255, 255, 255), thickness = 2)

            out.write(scaled)

        if args.verbose: print('Recorded frame {}'.format(nframe))

        nframe += 1


    ## Finalize all outputs -- video stream, detector, and tracker.
    if not args.no_output:

        out.release()
        detector.write(output.replace(".mp4", "_det.csv"))
        if tracker: tracker.write(output.replace(".mp4", "_tr.csv"))




if __name__ == '__main__': main()

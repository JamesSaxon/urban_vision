#!/usr/bin/env python

import os, sys

from tqdm import tqdm

import configargparse

import cv2

import numpy as np
import pandas as pd

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

import threading
import queue

import tracker as tr
import detector as det



def get_roi(vinput, scale, roi, roi_file = "", select_roi = False): 
    """
    This function checks takes its ROI from the first of four alternatives:
    1. `roi` -- a list of [xmin, xmax, ymin, ymax] explicitly indicated on the command line.
    2. An roi_file consisting of lines of video_filenames, xmin, xmax, ymin, ymax.
    3. If `select_roi` is specified and the video is valid, then use cv2.selectROI
    4. Failing all of these, then just use the frame width and height.
    Although the command-line options and file format take inputs as fractions,
    the returned ROIs are in number of pixels.
    """

    ## To get the frame dimensions and
    ## be able to call the cv2 ROI selector,
    ## we will open a video capture option.
    vid = cv2.VideoCapture(vinput)

    FRAMEX = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAMEY = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # If the path does not exist, overwrite options.
    if roi_file and not os.path.exists(roi_file): 
        roi_file, select_roi = "", True

    if not roi and roi_file:
        roi_df = pd.read_csv(roi_file, index_col = "file")

        # If the video is found, then rescale and return.
        # If it is not, default to the opencv method.
        if vinput in roi_df.index:
            ROI = roi_df.loc[vinput].to_dict()
            ROI = [int(ROI["xmin"] * FRAMEX), 
                   int(ROI["xmax"] * FRAMEX), 
                   int(ROI["ymin"] * FRAMEY),
                   int(ROI["ymax"] * FRAMEY)]

            vid.release()

            return ROI

        select_roi = True


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

    # No video file has been found, and we were neither _given_ an ROI
    # nor were we asked to specify one.  We revert to width x height.
    else: 

        ROI = [0, FRAMEX, 0, FRAMEY]

    vid.release()

    return ROI


global COMPLETE_INPUT, COMPLETE_DETECTION, COMPLETE_TRACKING
COMPLETE_INPUT, COMPLETE_DETECTION, COMPLETE_TRACKING = False, False, False

def detect_objects_in_frame(qdetector, qtracker, detector, 
                            DRAW_DETECTOR, SHOW_HEAT, HEAT_FRAC, HEAT_FADE, SCALE):
    """
    This function loops on the detection queue (of frames).
    It runs all detector-based functionality (detections, heat map, drawing)
    and then passes the detections to the tracker queue.
    The detector is passed as an argument, because the "actual" edgetpu
    detector needs to be on the main thread.
    If input is complete, then it completes all frames, and signals completion.
    """

    while not (COMPLETE_INPUT and qdetector.empty()):

        try: nframe, frame, scaled = qdetector.get(timeout = 0.1)
        except: continue

        detections = detector.detect(frame)

        if SHOW_HEAT: 

            update = not(nframe % 10)
            heat_level = HEAT_FRAC * min(1, nframe / HEAT_FADE)
            scaled = detector.draw_heatmap(scaled, heat_level, update = update, scale = SCALE, xmin = 50)

        if DRAW_DETECTOR: scaled = detector.draw(scaled, scale = SCALE)

        qtracker.put((nframe, frame, scaled, detections))

        qdetector.task_done()

    global COMPLETE_DETECTION
    COMPLETE_DETECTION = True


def track_objects_in_frame(qtracker, qvideo = None, tracker = None, SCALE = 1):
    """
    The tracker receives frames and detections from the detector queue.
    If tracking is activated, it applies tracking and any drawing functionality.
    If an output video stream is provided, 
    it sends the drawn / decorated frame to that queue.
    When no frames remain, and detection is complete, it terminates / task_done.
    """

    while not (COMPLETE_DETECTION and qtracker.empty()):

        try: nframe, frame, scaled, detections = qtracker.get(timeout = 0.1)
        except: continue

        ##  Tracker only.
        if tracker:

            tracker.update(detections, frame)

            scaled = tracker.draw(scaled, scale = SCALE)

        if qvideo: qvideo.put((nframe, scaled))

        qtracker.task_done()

    global COMPLETE_TRACKING
    COMPLETE_TRACKING = True


def write_frame_to_stream(video_queue, 
                          VIDEO_OUT, FRAMEXS, FRAMEYS,
                          XMINS = None, XMAXS = None, YMINS = None, YMAXS = None, pretty_video = False):
    """
    This function simply writes frames to an output VideoWriter.
    It loops on the video_queue for inputs.
    If a scaled ROI is specified (XMAXS, etc.), it either draws a box or, 
    for `pretty_video`, shades the area outside the ROI.
    This is "pretty expensive", so I dod this only for "publicity."
    """

    VIDEO_OUT = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), 30, (FRAMEXS, FRAMEYS))

    shade = None
    if pretty_video:

        shade = 1.5 * np.ones((FRAMEYS, FRAMEXS)) # .astype("uint8")
        shade[YMINS:YMAXS,XMINS:XMAXS] = 1


    while not (COMPLETE_TRACKING and video_queue.empty()):

        try: nframe, frame = video_queue.get(timeout = 0.1)
        except: continue

        ##  Cut out the detection region.
        if XMINS is not None:

            ##  Either shade the unused portion, or just draw a rectangle.
            if pretty_video: frame = (frame / shade[:,:,np.newaxis]).astype("uint8")
            else: frame = cv2.rectangle(frame, tuple((XMINS, YMINS)), tuple((XMAXS, YMAXS)), (0, 0, 0), 3)

        ##  Write it, if the stream has not been turned off.
        frame = cv2.putText(frame, "{:05d}".format(nframe),
                            org = (int(FRAMEXS*0.02), int(FRAMEYS*0.10)),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2 if FRAMEXS > 600 else 1,
                            color = (255, 255, 255), thickness = 2)

        VIDEO_OUT.write(frame)

        video_queue.task_done()

    VIDEO_OUT.release()


def main(vinput, output, odir,
         nframes, nskip,
         yolo, model, labels, thresh, categs, max_det_items,
         roi, xgrid, ygrid, roi_loc, max_overlap, min_area, edge_veto, roi_buffer,
         no_tracker, match_method, max_missing, max_distance, min_distance_or,
         predict_matches, kalman_track, max_track, candidate_obs,
         draw_detector, kalman_viz, contrail, scale, view, no_output, no_video_output, pretty_video, 
         show_heat, heat_frac, heat_fade, geometry, verbose, 
         roi_file, select_roi, config):
    """
    Here, we build and configure the detector and tracker objects, and 
    build and collect the queues for detection, tracking, and video outputs.
    The detector is also creaed on the "main" thread.
    The frame reader is on the main thread, because I had "issues" otherwise.
    So "main" could be considered the 0th thread of the program.
    """


    ##  Open the input video file...
    vid = cv2.VideoCapture(vinput)
    
    FRAMEX, FRAMEY = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ##  Shape parameters.
    ROI = roi
    XMIN,  XMAX,  YMIN,  YMAX  = ROI
    XMINS, XMAXS, YMINS, YMAXS = [round(v/scale) for v in ROI]


    ##  And the output video file.
    video_out = None
    if not no_video_output: video_out = output + "_det.mp4"

    ##  Build the detector, including the TF engine.
    detector = det.Detector(yolo = yolo, model = model, labels = labels, categs = categs, thresh = thresh, k = max_det_items,
                            max_overlap = max_overlap, loc = roi_loc, edge_veto = edge_veto, min_area = min_area, verbose = False)

    detector.set_roi({"xmin" : XMIN, "xmax" : XMAX, "ymin" : YMIN, "ymax" : YMAX})
    detector.set_xgrid(xgrid)
    detector.set_ygrid(ygrid)

    if geometry and show_heat: detector.set_world_geometry(geometry, scale = scale)


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

    draw_frames = bool(view or not no_video_output)

    # Start the threads -- 
    detect_queue = queue.Queue(50)
    track_queue  = queue.Queue(50)
    video_queue  = queue.Queue(50) if draw_frames else None

    detect_thread = threading.Thread(target = detect_objects_in_frame,
                                     args = (detect_queue, track_queue, detector, 
                                             draw_detector, show_heat and draw_frames,
                                             heat_frac, heat_fade, scale))
    detect_thread.start()

    track_thread = threading.Thread(target = track_objects_in_frame, 
                                    args = (track_queue, video_queue, tracker, scale))
    track_thread.start()


    if video_queue:

        video_thread = threading.Thread(target = write_frame_to_stream, 
                                        args = (video_queue, video_out, 
                                                round(FRAMEX / scale), round(FRAMEY / scale), 
                                                XMINS, XMAXS, YMINS, YMAXS, pretty_video))
        video_thread.start()


    nframe = 0
    pbar = tqdm(desc = args.vinput, total = nframes)
    while True:

        ret, frame = vid.read()

        if not ret or nframe >= nframes:
            break

        if nframe < nskip:
            nframe += 1
            continue

        ##  Useful throughout...
        ##    for reasons *completely* beyond me, this line actually 
        ##    speeds up the code by about a factor of two, 
        ##  so I'm doing it even when there is no output
        scaled = cv2.resize(frame, None, fx = 1 / scale, fy = 1 / scale)

        ## The detect queue will deliver the frame 
        ##  to the tracking and output queues.
        detect_queue.put((nframe, frame, scaled))

        nframe += 1
        pbar.update()

    pbar.close()


    # The inputs are now complete.
    # Set a global flag, so that the other threads will terminate
    #  when they complete all frames that remain in their queues.
    global COMPLETE_INPUT
    COMPLETE_INPUT = True
    detect_thread.join()
    track_thread.join()


    ## Finalize all outputs -- video stream, detector, and tracker.
    if not no_video_output:

        video_thread.join()

    # Unless told not to, let's write CSV output files.
    if not no_output:

        detector.write(output + "_det.csv")
        if tracker: tracker.write(output + "_tr.csv")



if __name__ == '__main__':

    parser = configargparse.ArgParser(default_config_files = ['./stream_defaults.conf'])
    parser.add('-c', '--config', required = False, is_config_file = True, help = 'Path for config file.')

    ## Basic inputs
    parser.add('-i', '--vinput', help = 'Directory of input images.', required=True)
    parser.add('-o', '--output', help = 'Name of output file, by default derived from input name.', default = "")
    parser.add('--odir',         help = 'Name of output directory, by default derived from input name.', default = "")

    ## What to process
    parser.add("-f", "--nframes", default = float("inf"), type = float, help = "Total number of frames to process")
    parser.add("--nskip", default = 0, type = int, help = "Skip deeper into a stream (useful for debugging)")

    ## Model / Detection Engine
    parser.add('-m', '--model', required = True, help = "The tflite model")
    parser.add('--labels', required = True, help='Path of the labels file.')
    parser.add('--thresh', default = 0.4, type = float, help = "Confidence level of detections")
    parser.add("--categs", default = ["car"], nargs = "+", help = "Types of objects to process -- must match labels")
    parser.add('--max_det_items', default = 50, type = int, help = "'k' parameter of max detections, for engine.")

    ## Detector Parameters
    parser.add("--yolo", default = False, action = "store_true", help = "Switch to YOLO detector from SSD (SSD is the default).")
    parser.add("--select_roi", default = False, action = "store_true", help = "Interactively re-select the ROI.")
    parser.add("--roi_file", default = "", type = str, help = "Get the ROI for this file from a csv file (if it can be found).")
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

    parser.add("--draw_detector", default = False, action = "store_true", help = configargparse.SUPPRESS)

    parser.add("--show_heat", action = "store_true", default = False, help = "Whether to show a projected geometry heat map.")
    parser.add("--heat_frac", type = float, default = 0.5, help = "Max alpha value")
    parser.add("--heat_fade", type = int, default = 0, help = "Number of frames, to 'fade in' geometry")
    parser.add("--geometry", type = str, default = "", help = "Filename of geometry, to calculate projection matrix.")

    parser.add("-v", "--verbose", default = False, action = "store_true", help = "")

    args = parser.parse_args()

    if args.verbose: print(parser.format_values()) ## Where did the settings come from!?

    if args.no_output: args.no_video_output = True

    if not args.output: # if it's not defined, get it from the input file.

        if not args.odir: output = args.vinput
        else: output = args.odir + "/" + args.vinput.split("/")[-1]

        for f in [".gif", ".mov", ".MOV", ".mp4"]: output = output.replace(f, "")
        args.output = output

    if args.no_tracker and (not args.no_video_output or args.view): args.draw_detector = True
    else: args.draw_detector = False

    if args.nframes <= 0: args.nframes = float("inf")
    # If only cv2.CAP_PROP_FRAME_COUNT or cv2.CAP_PROP_FRAME_COUNT were reliable!

    args.roi = get_roi(args.vinput, args.scale, args.roi, args.roi_file, args.select_roi)

    main(**vars(args))




import os

import cv2

import numpy as np
import pandas as pd

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from scipy.ndimage import gaussian_filter

from PIL import Image



class BBox():
    """
    Convenience functions for a bounding box composed of 
    pixel extents xmin, xmax, ymin, ymax.
    """

    def __init__(self, xmin, xmax, ymin, ymax):

        self.xmin = min(xmin, xmax)
        self.xmax = max(xmin, xmax)
        self.ymin = min(ymin, ymax)
        self.ymax = max(ymin, ymax)

        self.area = (xmax - xmin) * (ymax - ymin)

    def __repr__(self):

        return "xmin={:.2f} xmax={:.2f} ymin={:.2f} ymax={:.2f}".format(self.xmin, self.xmax, self.ymin, self.ymax)

    def __str__(self):

        return self.__repr__()

    def min_and_width(self):
        
        return (self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin)

    def loc(self, hloc = None, vloc = None):

        x = (self.xmax + self.xmin) / 2
        if hloc == "left":   x = self.xmax
        if hloc == "right":  x = self.xmin

        y = (self.ymax + self.ymin) / 2
        if vloc == "upper":  y = self.ymin
        if vloc == "lower":  y = self.ymax

        return x, y 

    def draw_rectangle(self, img, scale = 1, color = (255, 255, 255), width = 1):

        return cv2.rectangle(img, 
                             tuple((int(self.xmin / scale), int(self.ymin / scale))),
                             tuple((int(self.xmax / scale), int(self.ymax / scale))),
                             color, width)

    def intersection(self, other):
    
        # coordinates of the intersection rectangle
        xmin = max(self.xmin, other.xmin)
        xmax = min(self.xmax, other.xmax)
        ymin = max(self.ymin, other.ymin)
        ymax = min(self.ymax, other.ymax)

        if xmin > xmax or ymin > ymax:
            return None
    
        return BBox(xmin, xmax, ymin, ymax)

    def intersection_area(self, other):
    
        # coordinates of the intersection rectangle
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
    
        # compute the area of intersection rectangle
        return max(0, xmax - xmin) * max(0, ymax - ymin)
    
    
    def max_fractional_intersection(self, other):
    
        intx_area = self.intersection_area(other)
    
        return max(intx_area / self.area, intx_area / other.area)



class Detection():
    """
    A detection consists of a bounding box along with a confidence value and a label.
    We can also specify a "point" on the box (upper, middle, lower; left, center, right)
      as the default location for tracking.
    These values are protected in Detector() instead of here.
    And just for fun, we can set the color.
    """

    colors = {"person" : (0, 0, 255),
              "car" : (0, 255, 255), "bus" : (0, 255), "truck" : (255, 0, 0),
              "dog" : (255, 0, 255), "bike" : (125, 255, 255), "bicycle" : (125, 255, 255), 
              "stroller" : (125, 125, 255)}

    def __init__(self, box = None, conf = None, label = None, frame_id = None, color = None):

        if box is None: 
            self.box  = None
            self.area = 1.0
        else:
            self.box = BBox(**box)
            self.area = self.box.area

        self.xy = self.box.loc()
        self.x = self.xy[0]
        self.y = self.xy[1]

        self.hloc = "center"
        self.vloc = "middle"

        self.conf  = conf
        self.label = label

        self.frame = frame_id

        if   color is not None: self.color = color
        elif label is not None: self.color = Detection.colors[label]
        else: self.color = None


    def set_reference_location(self, hloc = None, vloc = None):

        self.xy = self.box.loc(hloc, vloc)
        self.x = self.xy[0]
        self.y = self.xy[1]

        self.hloc = hloc
        self.vloc = vloc


    def bbox_intersection(self, other):
    
        return self.box.intersection(other.box)
    
    def max_fractional_intersection(self, other):
    
        return self.box.max_fractional_intersection(other.box)

    def set_bbox(self, new_box):

        self.box = new_box
        self.xy = self.box.loc(self.hloc, self.vloc)
        self.x = self.xy[0]
        self.y = self.xy[1]

        self.area = self.box.area



class Detector():
    """
    Detector is, primarily, a wrapper for an edgetpu SSD detector or a YOLO detector.
    This entails labels, confidence levels, ROIs (potentially gridded, for the SSD.
    All detections are saved over time; individual detections can be visualized as bounding boxes
      and cumulative patterns can be shown as a heatmap.
    In addition, there is some functionality for transforming detections from the camera frame
    into a real-world geometry, i.e., generating the homography from a series of control points.
    These are specified in a geography file -- see set_world_geometry.
    """

    colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]

    def __init__(self,
                 model = "../models/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite",
                 labels = "../models/coco_labels.txt",
                 relative_coord = False, keep_aspect_ratio = False,
                 categs = ["person"], thresh = 0.6, k = 1,
                 max_overlap = 0.5, min_area = 0,
                 loc = "upper center", edge_veto = 0,
                 yolo_path = "", yolo_size = 0, nms_thresh = 0.3,
                 static_detections = "",
                 verbose = False):


        self.labels = dataset_utils.read_label_file(labels) if labels else None

        self.yolo = (yolo_size > 0) and yolo_path
        if self.yolo:

            self.yolo_size = yolo_size

            yolo_path    = os.path.abspath(yolo_path)
            yolo_config  = yolo_path + "/cfg"
            yolo_weights = yolo_path + "/wgts"
            yolo_labels  = yolo_path + "/names"
            
            self.yolo_model = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
            self.yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            yn = self.yolo_model.getLayerNames()
            self.yolo_names = [yn[i[0] - 1] for i in self.yolo_model.getUnconnectedOutLayers()]

            self.yolo_labels = open(yolo_labels).read().strip().split("\n")

            self.nms_thresh = nms_thresh

        else: 

            self.engine = DetectionEngine(model)

        if static_detections:

            self.static_detections = pd.read_csv(static_detections)

        else: self.static_detections = None

        self.categs = categs
        self.ncategs = {li for li, l in self.labels.items() if l in categs}

        self.thresh = thresh
        self.k      = k

        self.roi = None
        self.roi_bbox = None
        self.xgrid = 1
        self.ygrid = 1

        self.overlap = 0.1 ## In the grids...

        self.max_overlap = max_overlap
        self.min_area = min_area

        self.relative_coord = relative_coord
        self.keep_aspect_ratio = keep_aspect_ratio


        loc = loc.split(" ")
        self.vloc, self.hloc = loc[0], loc[1]

        if self.vloc not in ["upper", "middle", "lower"]:
            raise(ValueError, "Vertical location must be upper, middle, or lower.")

        if self.hloc not in ["left", "center", "right"]:
            raise(ValueError, "Horizontal location must be left, center, or right.")

        self.edge_veto = edge_veto

        self.verbose = verbose

        self.frame = 0

        self.detections = []
        self.all_detections = []

        self.has_world_geometry = False

    def set_xgrid(self, xgrid):

        self.xgrid = xgrid

    def set_ygrid(self, ygrid):

        self.ygrid = ygrid

    def set_roi(self, roi):

        self.roi = roi
        self.xmin = roi["xmin"]
        self.xmax = roi["xmax"]
        self.ymin = roi["ymin"]
        self.ymax = roi["ymax"]

        self.roi_bbox = BBox(self.xmin, self.xmax, self.ymin, self.ymax)


    def set_world_geometry(self, geofile, scale = 1, inv_binsize = 10):

        self.has_world_geometry = True

        self.inv_binsize = inv_binsize

        localized = pd.read_csv(geofile)

        localized[["xp", "yp"]] = localized[["xp", "yp"]].astype(float)

        self.geo_xmin, self.geo_xmax = localized.x.min(), localized.x.max()
        self.geo_ymin, self.geo_ymax = localized.y.min(), localized.y.max()

        self.geo_xrange = self.geo_xmax - self.geo_xmin
        self.geo_yrange = self.geo_ymax - self.geo_ymin

        localized.dropna(inplace = True)

        localized["x"] -= self.geo_xmin
        localized["y"] -= self.geo_ymin

        src     = localized[["xp", "yp"]].values
        dst     = localized[["x",  "y" ]].values 
        dst_inv = localized[["x",  "y" ]].values * inv_binsize

        self.geo_homography = cv2.findHomography(src, dst)[0]
        self.geo_inv_homography = np.linalg.pinv(cv2.findHomography(src / scale, dst_inv)[0])

        self.SROI = [int(v/scale) for v in [self.xmin, self.xmax, self.ymin, self.ymax]]


    def remove_duplicates(self, match_labels = False):
    
        duplicates = []
        for io, iobj in enumerate(self.detections):
    
            for jo, jobj in enumerate(self.detections[io+1:]):
    
                jo += io + 1
    
                if match_labels and (iobj.label != jobj.label): continue
    
                intx_frac = iobj.max_fractional_intersection(jobj)
    
                if intx_frac > self.max_overlap:
    
                    duplicates.append(io if iobj.conf < jobj.conf else jo)
    
        
        for d in sorted(list(set(duplicates)), reverse = True): 
            self.detections.pop(d)
    



    def detect(self, frame, frame_id = None):
        """
        Run YOLO or SSD according to settings!
        """

        self.detections = []

        if self.static_detections is not None:
            self.detect_static(frame, frame_id)
        elif self.yolo:
            self.detect_yolo(frame, frame_id)
        else:
            self.detect_ssd(frame, frame_id)
        
        return self.detections

    def detect_static(self, frame, frame_id):

        if frame_id is None: 
            print("Frame ID must be specified for static detections.")
            return

        height, width = frame.shape[:2]

        query = f"(frame == {frame_id}) & (label in @self.categs) & (conf >= {self.thresh})"

        static_det_subset = self.static_detections.query(query)

        self.detections = []
        for di, det in static_det_subset.iterrows():

            box_dict = {"xmin" : int(det.xmin * width),  "xmax" : int(det.xmax * width),
                        "ymin" : int(det.ymin * height), "ymax" : int(det.ymax * height) }
        
            det = Detection(box_dict, det.conf, det.label, frame_id)

            if self.roi_bbox:

                intx = det.box.intersection(self.roi_bbox)
                if intx is None: continue

                det.set_bbox(intx)

            if det.area / height / width < self.min_area: continue

            det.set_reference_location(self.hloc, self.vloc)

            self.detections.append(det)

        self.all_detections.extend(self.detections)


    def detect_yolo(self, frame, frame_id = None):
        """
        Pretty straightforward application of the cv2.dnn.
        All of the work is in configuring the GPU!!
        """

        self.frame += 1
        if frame_id is None: frame_id = self.frame 

        if self.roi: 
            roi_xmin = self.xmin
            roi_xmax = self.xmax
            roi_ymin = self.ymin
            roi_ymax = self.ymax

        else:
            roi_xmin, roi_ymin = 0, 0
            roi_ymax, roi_xmax = frame.shape[:2]

        height, width = frame.shape[:2]

        range_x = roi_xmax - roi_xmin
        range_y = roi_ymax - roi_ymin

        frame_roi = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        blob = cv2.dnn.blobFromImage(frame_roi, 1 / 255, (self.yolo_size, self.yolo_size),
                                     swapRB = True, crop = False)

        self.yolo_model.setInput(blob)
        
        yolo_detections = self.yolo_model.forward(self.yolo_names)
        
        boxes, confs, classes = [], [], []
        
        # loop over each of the layer outputs
        for output in yolo_detections:
        
            # loop over each of the detections
            for det in output:
            
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                
                # filter weak detections.
                if conf < self.thresh: continue
                
                # YOLO returns center + dimensions.
                # scale the bbox to to image size
                box = det[0:4] * np.array([range_x, range_y, range_x, range_y])
                (ctr_x, ctr_y, width, height) = box.astype("int")
                    
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(ctr_x - (width  / 2)) + roi_xmin
                y = int(ctr_y - (height / 2)) + roi_ymin
                    
                # Update lists needed for NMS.
                boxes  .append([x, y, int(width), int(height)])
                confs  .append(float(conf))
                classes.append(class_id)


        self.detections = []

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.thresh, self.nms_thresh)

        if not len(idxs): return self.detections
        else: idxs = set(idxs.flatten())
        
        for i in idxs:
        
            label = self.yolo_labels[classes[i]]

            # print('detection!', x, y, confs[i], class_id, label, self.categs)
            if label not in self.categs: continue
            
            box_dict = {"xmin" : boxes[i][0], "xmax" : boxes[i][0] + boxes[i][2], 
                        "ymin" : boxes[i][1], "ymax" : boxes[i][1] + boxes[i][3]}
        
            det = Detection(box_dict, confs[i], label, frame_id)
            
            if det.area / height / width < self.min_area: continue

            det.set_reference_location(self.hloc, self.vloc)
            self.detections.append(det)
            
        self.all_detections.extend(self.detections)

        return self.detections


    def detect_ssd(self, frame, frame_id = None):
        """
        The short version of this is that it is just applying the edgetput detector.
        The slightly longer version is that we can also "grid" the SSD detection area.
        """

        self.frame += 1
        if frame_id is None: frame_id = self.frame 

        height, width = frame.shape[:2]

        if self.roi: 
            roi_xmin = self.xmin
            roi_xmax = self.xmax
            roi_ymin = self.ymin
            roi_ymax = self.ymax
        else:
            roi_xmin, roi_ymin = 0, 0
            roi_ymax, roi_xmax = frame.shape[:2]

        range_x = roi_xmax - roi_xmin
        range_y = roi_ymax - roi_ymin

        # ROI list from grid.
        xvals = np.linspace(roi_xmin, roi_xmax, self.xgrid+1)
        yvals = np.linspace(roi_ymin, roi_ymax, self.ygrid+1)

        # Set number of overlap pixels
        if self.xgrid != 1:
            xoverlap = int(self.overlap*(roi_xmax-roi_xmin)/self.xgrid)
        else: xoverlap = 0

        if self.ygrid != 1:
            yoverlap = int(self.overlap*(roi_ymax-roi_ymin)/self.ygrid)
        else: yoverlap = 0

        subroi_list = []
        for i in range(self.xgrid):
            for j in range(self.ygrid):

                xmax = int(xvals[i+1])
                ymax = int(yvals[j+1])

                if i == 0: xmin = int(xvals[i])
                else:      xmin = int(xvals[i] - xoverlap)

                if j == 0: ymin = int(yvals[j])
                else:      ymin = int(yvals[j] - yoverlap)

                subroi_list.append([xmin, xmax, ymin, ymax])

        self.detections = []

        for subroi in subroi_list:

            subroi_xmin, subroi_xmax, subroi_ymin, subroi_ymax = subroi
            subroi_range_x = subroi_xmax - subroi_xmin 
            subroi_range_y = subroi_ymax - subroi_ymin 

            subimage = Image.fromarray(cv2.cvtColor(frame[subroi_ymin:subroi_ymax,
                                                       subroi_xmin:subroi_xmax],
                                                       cv2.COLOR_BGR2RGB))

            raw_detections = self.engine.detect_with_image(subimage, threshold = self.thresh,
                                                           keep_aspect_ratio = False, 
                                                           relative_coord = False, top_k = self.k)

            # print(len(raw_detections), "detections")
            for iobj, obj in enumerate(raw_detections):

                # If the label is irrelevant, just get out.
                label = None
                # print(obj.score, obj.label_id, obj.bounding_box.flatten())
                if self.labels is not None and len(self.categs):

                    if obj.label_id not in self.ncategs: continue
                    label = self.labels[obj.label_id]

                box = obj.bounding_box.flatten()

                if self.edge_veto > 0:
                    if box[0] / subroi_range_x < self.edge_veto:     continue
                    if box[2] / subroi_range_x > 1 - self.edge_veto: continue
                    if box[1] / subroi_range_y < self.edge_veto:     continue
                    if box[3] / subroi_range_y > 1 - self.edge_veto: continue

                box[0] += subroi_xmin
                box[2] += subroi_xmin
                box[1] += subroi_ymin
                box[3] += subroi_ymin

                # draw_box = box.astype(int)

                box_xmin, box_ymin, box_xmax, box_ymax = box
                box_dict = {"xmin" : box_xmin, "xmax" : box_xmax, "ymin" : box_ymin, "ymax" : box_ymax}

                det = Detection(box_dict, obj.score, label, frame_id)

                if det.area / height / width < self.min_area: continue

                det.set_reference_location(self.hloc, self.vloc)

                self.detections.append(det)


        self.remove_duplicates()

        self.all_detections.extend(self.detections)

        return self.detections



    def draw(self, frame, scale, width = 1, color = (255, 255, 255)):
        """
        Iterate over current detections and draw their bounding boxes.
        """

        for d in self.detections:

            d.box.draw_rectangle(frame, scale, color = (255, 255, 255), width = 1)

        return frame


    def write(self, file_name):
        """
        Write a CSV file of all detections.
        """
    
        df = pd.DataFrame([{"frame": d.frame, "x" : d.x, "y" : d.y, "area" : d.area, "conf" : d.conf, "label" : d.label,
                            "xmin" : d.box.xmin, "xmax" : d.box.xmax, "ymin" : d.box.ymin, "ymax" : d.box.ymax}
                           for d in self.all_detections])

        if not df.shape[0]: 
            print("No frames with detections; not writing to file.")
            return


        df = df[["frame", "conf", "label", # "x", "y", "area",
                 "xmin", "xmax", "ymin", "ymax"]]

        df.sort_values(by = ["frame", "conf"]).to_csv(file_name, float_format = "%.3f", header = True, index = False)


    def naive_heatmap(self, size, scale, blur, quantile, cmap, xmin = None):

        img = np.zeros(size)

        xy = np.array([[det.x / scale, det.y / scale] 
                       for det in self.all_detections]).astype(int)
        
        for x, y in xy:

            if xmin is not None and x < xmin: continue

            img[y, x] += 1
        
        img8 = np.where(img > np.quantile(img, quantile), 255, 
                        255 * img / img.max()).astype("uint8")

        img8_blur = cv2.GaussianBlur(img8, (blur, blur), 0)
        img_col = cv2.applyColorMap(img8_blur, cmap)

        mask = np.ones(size).astype(bool)
        
        return (mask, img_col)
        

    def projected_heatmap(self, size, scale, blur, quantile, cmap):
        """
        Same as above, but the heatmap is calculated in a transformed geometry
        so that each detection area corresponds to "ground" area instead of pixels.
        Regions outside of the bounding box of the mapped area are simply masked off.
        """

        if not self.all_detections: 
            return np.ones(size).astype(bool), np.zeros((size[0], size[1], 3)).astype("uint8")

        xy = np.array([[det.x, det.y] for det in self.all_detections])[np.newaxis]

        xyW = cv2.perspectiveTransform(xy, self.geo_homography).reshape(-1, 2)
        
        img = np.zeros((int(self.geo_yrange * self.inv_binsize),
                        int(self.geo_xrange * self.inv_binsize)))
        
        for x, y in (self.inv_binsize * xyW).astype(int):
            
            if x < 0: continue
            if y < 0: continue
            if x >= img.shape[1]: continue
            if y >= img.shape[0]: continue
            
            img[y,x] += 1
            
        img = gaussian_filter(img, blur)
        
        max_val = max(np.quantile(img, quantile), 1)
        img8 = np.where(img > max_val, 255, 255. * img / max_val).astype("uint8")
        
        img8_col = cv2.applyColorMap(img8, cv2.COLORMAP_PARULA)
        img8_col = cv2.warpPerspective(img8_col, self.geo_inv_homography, dsize = (size[1], size[0]))

        mask = np.ones((int(self.geo_yrange * self.inv_binsize),
                        int(self.geo_xrange * self.inv_binsize))) * 255

        mask = cv2.warpPerspective(mask.astype("uint8"), self.geo_inv_homography,
                                   dsize = (size[1], size[0]))
        mask = mask > 0

        return mask, img8_col

        

    def draw_heatmap(self, frame, heat_level = 0.5, update = True, scale = 1,
                     blur = 1, quantile = 0.99, cmap = cv2.COLORMAP_PARULA, xmin = 0):
        """
        Draw the naive or projected heatmap, based on whatever is available!
        """

        size = frame.shape[:2]

        XMINS, XMAXS, YMINS, YMAXS = self.SROI

        if update:

            if not self.has_world_geometry: 
                self.mask, self.heat = self.naive_heatmap(size, scale, blur, quantile, cmap, xmin)
            else: 
                self.mask, self.heat = self.projected_heatmap(size, scale, blur, quantile, cmap)


        if not self.roi:
            frame[self.mask] = (frame[self.mask] * (1 - heat_level) + heat[self.mask] * heat_level).astype("uint8")

        else:

            smask_roi = self.mask[YMINS:YMAXS,XMINS:XMAXS]

            frame[YMINS:YMAXS,XMINS:XMAXS][smask_roi] = \
                (frame[YMINS:YMAXS,XMINS:XMAXS][smask_roi] * (1 - heat_level) + \
                 self.heat[YMINS:YMAXS,XMINS:XMAXS][smask_roi] * heat_level).astype("uint8")

        return frame
        


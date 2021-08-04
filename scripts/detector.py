import cv2, numpy as np

import pandas as pd

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from scipy.ndimage import gaussian_filter

from PIL import Image


def bbox_intersection(A, B):

    # coordinates of the intersection rectangle
    xmin = max(A.xmin, B.xmin)
    ymin = max(A.ymin, B.ymin)
    xmax = min(A.xmax, B.xmax)
    ymax = min(A.ymax, B.ymax)

    # compute the area of intersection rectangle
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)

    return intersection

def max_fractional_intersection(objA, objB):

    intx_area = bbox_intersection(objA.box, objB.box)

    return max(intx_area / objA.area, intx_area / objB.area)


def remove_duplicates(detections, max_overlap = 0.25, match_labels = False, labels = None):

    duplicates = []
    for io, iobj in enumerate(detections):

        for jo, jobj in enumerate(detections[io+1:]):

            jo += io + 1

            if match_labels and (iobj.label != jobj.label): continue

            intx_frac = max_fractional_intersection(iobj, jobj)

            if intx_frac > max_overlap:

                duplicates.append(io if iobj.conf < jobj.conf else jo)

    
    for d in sorted(list(set(duplicates)), reverse = True): 
        detections.pop(d)

    return 


class BBox():

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

    def loc(self, hloc, vloc):

        if hloc == "left":   x = self.xmax
        if hloc == "center": x = (self.xmax + self.xmin) / 2
        if hloc == "right":  x = self.xmin

        if vloc == "upper":  y = self.ymin
        if vloc == "middle": y = (self.ymax + self.ymin) / 2
        if vloc == "lower":  y = self.ymax

        return x, y 

    def draw_rectangle(self, img, scale = 1, color = (255, 255, 255), width = 1):

        return cv2.rectangle(img, 
                             tuple((int(self.xmin / scale), int(self.ymin / scale))),
                             tuple((int(self.xmax / scale), int(self.ymax / scale))),
                             color, width)



class Detection():

    colors = {"person" : (0, 0, 255),
              "car" : (0, 255, 255), "bus" : (0, 255), "truck" : (255, 0, 0),
              "dog" : (255, 0, 255), "bike" : (125, 255, 255), "bicycle" : (125, 255, 255), 
              "stroller" : (125, 125, 255)}

    def __init__(self, xy, box = None, conf = None, label = None, frame_id = None, color = None):

        self.xy    = xy
        self.x     = xy[0]
        self.y     = xy[1]

        if box is None: 
            self.box  = None
            self.area = 1.0
        else:
            self.box = BBox(**box)
            self.area = self.box.area

        self.conf  = conf
        self.label = label

        self.frame = frame_id

        if   color is not None: self.color = color
        elif label is not None: self.color = Detection.colors[label]
        else: self.color = None



class Detector():

    colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]

    def __init__(self,
                 model = "../models/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite",
                 labels = "../models/coco_labels.txt",
                 relative_coord = False, keep_aspect_ratio = False,
                 categs = ["person"], thresh = 0.6, k = 1,
                 max_overlap = 0.5, min_area = 0,
                 loc = "upper center", edge_veto = 0, verbose = False):


        self.engine = DetectionEngine(model)
        self.labels = dataset_utils.read_label_file(labels) if labels else None

        self.categs = categs
        self.ncategs = {li for li, l in self.labels.items() if l in categs}

        self.thresh = thresh
        self.k      = k

        self.roi = None
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


    def detect_grid(self, frame, frame_id = None):

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

        range_x = roi_xmax - roi_xmin
        range_y = roi_ymax - roi_ymin

        #ROI list from grid.
        xvals = np.linspace(roi_xmin, roi_xmax, self.xgrid+1)
        yvals = np.linspace(roi_ymin, roi_ymax, self.ygrid+1)

        #Set number of overlap pixels
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
                                                keep_aspect_ratio=False, relative_coord=False, top_k=self.k)

            for iobj, obj in enumerate(raw_detections):

                # If the label is irrelevant, just get out.
                label = None
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

                draw_box = box.astype(int)

                box_xmin, box_ymin, box_xmax, box_ymax = box

                if self.hloc == "left":   x = box_xmax
                if self.hloc == "center": x = (box_xmax + box_xmin) / 2
                if self.hloc == "right":  x = box_xmin

                if self.vloc == "upper":  y = box_ymin
                if self.vloc == "middle": y = (box_ymax + box_ymin) / 2
                if self.vloc == "lower":  y = box_ymax

                box_dict = {"xmin" : box_xmin, "xmax" : box_xmax, "ymin" : box_ymin, "ymax" : box_ymax}

                det = Detection((x,y), box_dict, obj.score, label, frame_id)
                if det.area > self.min_area: self.detections.append(det)


        remove_duplicates(self.detections, labels = self.ncategs, max_overlap = self.max_overlap)

        self.all_detections.extend(self.detections)

        return self.detections



    def draw(self, frame, scale, width = 1, color = (255, 255, 255)):

        for d in self.detections:

            d.box.draw_rectangle(frame, scale, color = (255, 255, 255), width = 1)

        return frame


    def write(self, file_name):
    
        df = pd.DataFrame([{"frame": d.frame, "x" : d.x, "y" : d.y, "area" : d.area, "conf" : d.conf, "label" : d.label,
                            "xmin" : d.box.xmin, "xmax" : d.box.xmax, "ymin" : d.box.ymin, "ymax" : d.box.ymax}
                           for d in self.all_detections])

        df = df[["frame", "conf", "label", "x", "y", "area", "xmin", "xmax", "ymin", "ymax"]]

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

        # print(self.geo_yrange * self.inv_binsize)
        mask = np.ones((int(self.geo_yrange * self.inv_binsize),
                        int(self.geo_xrange * self.inv_binsize))) * 255

        mask = cv2.warpPerspective(mask.astype("uint8"), self.geo_inv_homography,
                                   dsize = (size[1], size[0]))
        mask = mask > 0

        return mask, img8_col

        

    def draw_heatmap(self, frame, heat_level = 0.5, update = True, scale = 1,
                     blur = 1, quantile = 0.99, cmap = cv2.COLORMAP_PARULA, xmin = 0):

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
        


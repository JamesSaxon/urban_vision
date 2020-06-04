import cv2, numpy as np

import pandas as pd

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from PIL import Image


def bbox_intersection(objA, objB):

    xminA, yminA, xmaxA, ymaxA = objA.bounding_box.flatten()
    xminB, yminB, xmaxB, ymaxB = objB.bounding_box.flatten()

    # coordinates of the intersection rectangle
    xmin = max(xminA, xminB)
    ymin = max(yminA, yminB)
    xmax = min(xmaxA, xmaxB)
    ymax = min(ymaxA, ymaxB)

    # compute the area of intersection rectangle
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)

    return intersection

def max_fractional_intersection(objA, objB):

    intx_area = bbox_intersection(objA, objB)

    xminA, yminA, xmaxA, ymaxA = objA.bounding_box.flatten()
    xminB, yminB, xmaxB, ymaxB = objB.bounding_box.flatten()

    area_A = (xmaxA - xminA) * (ymaxA - yminA)
    area_B = (xmaxB - xminB) * (ymaxB - yminB)

    return max(intx_area / area_A, intx_area / area_B)


def flag_duplicates(raw_detections, max_overlap = 0.25, labels = None):

    duplicates = []
    for io, iobj in enumerate(raw_detections):

        i_label = iobj.label_id
        if i_label in [2, 5, 7]: i_label = -1

        for jo, jobj in enumerate(raw_detections[io+1:]):

            jo += io + 1

            if labels is not None and iobj.label_id not in labels: continue

            j_label = iobj.label_id
            if j_label in [2, 5, 7]: j_label = -1

            if i_label != j_label: continue

            intx_frac = max_fractional_intersection(iobj, jobj)

            if intx_frac > max_overlap:

                duplicates.append(io if iobj.score < jobj.score else jo)

    return duplicates


def write(frame_id, det_list, file_name):
    out_df = pd.DataFrame(columns=['frame', 'x', 'y', 'xmin', 'ymin', 'xmax', 'ymax', 'area', 'label', 'conf'])
    x, y, xmin, ymin, xmax, ymax, area, label, conf = [], [], [], [], [], [], [], [], []
    for detection in det_list:
        x.append(detection.xy[0])
        y.append(detection.xy[1])
        xmin.append(detection.box.xmin)
        ymin.append(detection.box.ymin)
        xmax.append(detection.box.xmax)
        ymax.append(detection.box.ymax)
        area.append(detection.area)
        label.append(detection.label)
        conf.append(detection.conf)

    out_df['x'] = x
    out_df['y'] = y
    out_df['xmin'] = xmin
    out_df['ymin'] = ymin
    out_df['xmax'] = xmax
    out_df['ymax'] = ymax
    out_df['area'] = area
    out_df['label'] = label
    out_df['conf'] = conf
    out_df['frame'] = frame_id

    out_df.to_csv(file_name, header=False, mode='a')

class BBox():

    def __init__(self, xmin, xmax, ymin, ymax):

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.area = (xmax - xmin) * (ymax - ymin)

class Detection():

    colors = {"person" : (0, 0, 255), "car" : (0, 255, 0), "truck" : (0, 255, 255)}

    def __init__(self, xy, box = None, conf = None, label = None, color = None):

        self.xy    = xy
        self.x     = xy[0]
        self.y     = xy[1]

        if box is None: 
            self.box  = None
            self.area = None
        else:
            self.box = BBox(**box)
            self.area = self.box.area

        self.conf  = conf
        self.label = label

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
                 max_overlap = 0.5,
                 loc = "upper center", edge_veto = 0, verbose = False):


        print("Loading", model, labels)

        self.engine = DetectionEngine(model)
        self.labels = dataset_utils.read_label_file(labels) if labels else None

        self.categs = categs
        self.ncategs = {li for li, l in self.labels.items() if l in categs}

        self.thresh = thresh
        self.k      = k

        self.max_overlap = max_overlap

        self.relative_coord = relative_coord
        self.keep_aspect_ratio = keep_aspect_ratio

        self.roi    = None

        loc = loc.split(" ")
        self.vloc, self.hloc = loc[0], loc[1]

        if self.vloc not in ["upper", "middle", "lower"]:
            raise(ValueError, "Vertical location must be upper, middle, or lower.")

        if self.hloc not in ["left", "center", "right"]:
            raise(ValueError, "Horizontal location must be left, center, or right.")

        self.edge_veto = edge_veto

        self.verbose = verbose


    ##  def detect(self, img, color = None,
    ##             return_image = False, return_areas = False, return_confs = False):


    ##      tensor = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ##      raw_detections = self.engine.detect_with_image(tensor,
    ##                                          threshold = self.thresh, top_k = self.k,
    ##                                          keep_aspect_ratio = False, relative_coord = False)

    ##      duplicates = flag_duplicates(raw_detections, labels = self.ncategs)

    ##      XY, AREAS, CONFS = [], [], []

    ##      for iobj, obj in enumerate(raw_detections):

    ##          # If the labels file and category filter
    ##          # are both defined, then filter.
    ##          label = None
    ##          if self.labels is not None and len(self.categs):

    ##              if obj.label_id not in self.ncategs: continue
    ##              label = self.labels[obj.label_id]

    ##          is_duplicate = iobj in duplicates

    ##          if self.labels is not None and self.verbose:
    ##              print(self.labels[obj.label_id] + ",", end = " ")

    ##          if is_duplicate: continue

    ##          # Draw a rectangle.
    ##          box = obj.bounding_box.flatten()
    ##          xmin, ymin, xmax, ymax = box

    ##          if return_image:

    ##              if is_duplicate: color = tuple([int((c + 255)/2) for c in color])
    ##              width = 2 if is_duplicate else 4

    ##              cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 4)

    ##          if self.verbose:
    ##              print('conf. = ', obj.score)
    ##              print('-----------------------------------------')

    ##          if self.hloc == "left":   x = xmax
    ##          if self.hloc == "center": x = (xmax + xmin) / 2
    ##          if self.hloc == "right":  x = xmin

    ##          if self.vloc == "upper":  y = ymin
    ##          if self.vloc == "middle": y = (ymax + ymin) / 2
    ##          if self.vloc == "lower":  y = ymax

    ##          XY.append((x, y))
    ##          AREAS.append((xmax - xmin) * (ymax - ymin))
    ##          CONFS.append(obj.score)

    ##      retval = [np.array(XY)]

    ##      if return_areas : retval.append(np.array(AREAS))
    ##      if return_confs : retval.append(np.array(CONFS))
    ##      if return_image : retval.append(img)

    ##      return retval


    def detect(self, frame, roi = None, return_image = True):

        if roi: roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
        else:   roi_xmin, roi_ymin, roi_xmax, roi_ymax = 0, 0, frame.shape[1], frame.shape[0]

        range_x = roi_xmax - roi_xmin
        range_y = roi_ymax - roi_ymin

        image = Image.fromarray(cv2.cvtColor(frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax], cv2.COLOR_BGR2RGB))

        raw_detections = self.engine.detect_with_image(image, threshold = self.thresh,
                                            keep_aspect_ratio=False, relative_coord=False, top_k=self.k)

        duplicates = flag_duplicates(raw_detections, labels = self.ncategs, max_overlap = self.max_overlap)

        ## XY, BOXES, AREAS, CONFS, LABELS = [], [], [], [], []

        det_list = []

        for iobj, obj in enumerate(raw_detections):

            # If the label is irrelevant, just get out.
            label = None
            if self.labels is not None and len(self.categs):

                if obj.label_id not in self.ncategs: continue
                label = self.labels[obj.label_id]

            box = obj.bounding_box.flatten()
            print(box)

            if self.edge_veto > 0:
                if box[0] / range_x < self.edge_veto:     continue
                if box[2] / range_x > 1 - self.edge_veto: continue
                if box[1] / range_y < self.edge_veto:     continue
                if box[3] / range_y > 1 - self.edge_veto: continue


            box[0] += roi_xmin
            box[2] += roi_xmin
            box[1] += roi_ymin
            box[3] += roi_ymin
            draw_box = box.astype(int)

            is_duplicate = iobj in duplicates

            if return_image:

                color = Detector.colors[self.categs.index(label)] if len(self.categs) else (255, 255, 255)

                if is_duplicate: color = tuple([int((c + 255)/2) for c in color])
                width = 2 if is_duplicate else 4

                cv2.rectangle(frame, tuple(draw_box[:2]), tuple(draw_box[2:]), color, width)

            if is_duplicate: continue

            box_xmin, box_ymin, box_xmax, box_ymax = box

            if self.hloc == "left":   x = box_xmax
            if self.hloc == "center": x = (box_xmax + box_xmin) / 2
            if self.hloc == "right":  x = box_xmin

            if self.vloc == "upper":  y = box_ymin
            if self.vloc == "middle": y = (box_ymax + box_ymin) / 2
            if self.vloc == "lower":  y = box_ymax

            ## XY.append((x,y))
            ## LABELS.append(label)
            ## BOXES.append({"xmin" : box_xmin, "xmax" : box_xmax,
            ##               "ymin" : box_ymin, "ymax" : box_ymax})

            ## AREAS.append((box[2]-box[0])*(box[3]-box[1]))
            ## CONFS.append(obj.score)

            det_list.append(Detection((x,y), box_dict, obj.score, label))

            box_dict = {"xmin" : box_xmin, "xmax" : box_xmax, "ymin" : box_ymin, "ymax" : box_ymax}

            det_list.append(Detection((x,y), box_dict, obj.score, label))

        # retval = {"xy" : XY, "boxes" : BOXES, "areas" : AREAS, "confs" : CONFS, "labels": LABELS}
        if return_image: return det_list, frame

        return det_list


    def detect_grid(self, frame, roi = None, xgrid = 1, ygrid = 1, return_image = True, overlap = 0.1):

        if roi: roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
        else:   roi_xmin, roi_ymin, roi_xmax, roi_ymax = 0, 0, frame.shape[1], frame.shape[0]

        range_x = roi_xmax - roi_xmin
        range_y = roi_ymax - roi_ymin

        #ROI list from grid.
        xvals = np.linspace(roi_xmin, roi_xmax, xgrid+1)
        yvals = np.linspace(roi_ymin, roi_ymax, ygrid+1)

        #Set number of overlap pixels
        if xgrid != 1:
            xoverlap = int(overlap*(roi_xmax-roi_xmin)/xgrid)
        else: xoverlap = 0
        if ygrid != 1:
            yoverlap = int(overlap*(roi_ymax-roi_ymin)/ygrid)
        else: yoverlap = 0

        subroi_list = []
        for i in range(xgrid):
            for j in range(ygrid):
                xmax = int(xvals[i+1])
                ymax = int(yvals[j+1])
                if i == 0:
                    xmin = int(xvals[i])
                else:
                    xmin = int(xvals[i] - xoverlap)
                if j == 0:
                    ymin = int(yvals[j])
                else:
                    ymin = int(yvals[j] - yoverlap)

                subroi_list.append([xmin,
                                    xmax,
                                    ymin,
                                    ymax])

        ## XY, LABELS, BOXES, AREAS, CONFS = [], [], [], [], []

        det_list = []

        for subroi in subroi_list:
            subroi_xmin, subroi_xmax, subroi_ymin, subroi_ymax = subroi
            subimage = Image.fromarray(cv2.cvtColor(frame[subroi_ymin:subroi_ymax,
                                                       subroi_xmin:subroi_xmax],
                                                       cv2.COLOR_BGR2RGB))
            raw_detections = self.engine.detect_with_image(subimage, threshold = self.thresh,
                                                keep_aspect_ratio=False, relative_coord=False, top_k=self.k)

            duplicates = flag_duplicates(raw_detections, labels = self.ncategs, max_overlap = self.max_overlap)
            for iobj, obj in enumerate(raw_detections):

                # If the label is irrelevant, just get out.
                label = None
                if self.labels is not None and len(self.categs):

                    if obj.label_id not in self.ncategs: continue
                    label = self.labels[obj.label_id]

                box = obj.bounding_box.flatten()

                if self.edge_veto > 0:
                    if box[0] / range_x < self.edge_veto:     continue
                    if box[2] / range_x > 1 - self.edge_veto: continue
                    if box[1] / range_y < self.edge_veto:     continue
                    if box[3] / range_y > 1 - self.edge_veto: continue


                box[0] += subroi_xmin
                box[2] += subroi_xmin
                box[1] += subroi_ymin
                box[3] += subroi_ymin
                draw_box = box.astype(int)

                is_duplicate = iobj in duplicates

                if return_image:

                    color = Detector.colors[self.categs.index(label)] if len(self.categs) else (255, 255, 255)

                    if is_duplicate: color = tuple([int((c + 255)/2) for c in color])
                    width = 2 if is_duplicate else 4

                    cv2.rectangle(frame, tuple(draw_box[:2]), tuple(draw_box[2:]), color, width)

                if is_duplicate: continue

                box_xmin, box_ymin, box_xmax, box_ymax = box

                if self.hloc == "left":   x = box_xmax
                if self.hloc == "center": x = (box_xmax + box_xmin) / 2
                if self.hloc == "right":  x = box_xmin

                if self.vloc == "upper":  y = box_ymin
                if self.vloc == "middle": y = (box_ymax + box_ymin) / 2
                if self.vloc == "lower":  y = box_ymax


                ##  XY.append((x,y))
                ##  LABELS.append(label)
                ##  BOXES.append(box_dict)
                ##  AREAS.append((box[2]-box[0])*(box[3]-box[1]))
                ##  CONFS.append(obj.score)

                box_dict = {"xmin" : box_xmin, "xmax" : box_xmax, "ymin" : box_ymin, "ymax" : box_ymax}

                det_list.append(Detection((x,y), box_dict, obj.score, label))

        ## retval = {"xy" : XY, "boxes" : BOXES, "areas" : AREAS, "confs" : CONFS, "labels": LABELS}

        if return_image: return det_list, frame

        return det_list


    def detect_objects(self, frame, scale=4, kernel=60//4, panels=False,
                        box_size=300, top_k=3, view=False, gauss=True,
                        return_areas = False, return_confs = False):
        #Apply background subtraction
        if not kernel % 2: kernel +=1
        scaled = cv2.resize(frame, None, fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_AREA)
        mog = self.bkd.apply(scaled)


        mog[mog < 129] = 0
        if gauss: mog = cv2.GaussianBlur(mog, (kernel, kernel), 0)
        mog[mog < 50] = 0
        mog_mask = ((mog > 128) * 255).astype("uint8")

        #Find contours
        img = frame
        if panels: areas_inferenced = set()
        _, contours, _ = cv2.findContours(mog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        positions = []
        areas = []
        confs = []

        #Iterate through contours, set ROI, and detect
        contour_area_threshold = 300/scale
        for c in contours:
            if cv2.contourArea(c) < contour_area_threshold: continue
            #Calculate bounding box
            x, y, w, h = cv2.boundingRect(c)
            if panels:
                x_mid = (x + w//2)*scale
                y_mid = (y + h//2)*scale
                grid_square = (x_mid//(box_size - 50), y_mid//(box_size - 50))
                if grid_square in areas_inferenced: continue
                areas_inferenced.add(grid_square)
                xmin = max((box_size - 50)*grid_square[0] - 50, 0)
                xmax = xmin + box_size + 100
                ymin = max((box_size - 50)*grid_square[1] - 50, 0)
                ymax = ymin + box_size + 100
            else:
                width = max(box_size,w*2*scale)
                height = max(box_size,h*2*scale)
                x_mid = (x + w//2)*scale
                y_mid = (y + h//2)*scale
                xmin = max(0,x_mid - width//2)
                xmax = x_mid + width//2
                ymin = max(0,y_mid - height//2)
                ymax = y_mid + height//2
            roi = (xmin, ymin, xmax, ymax)
            #if view: cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

            #Run inference
            if view:
                pos, ar, con, img = self.detect_roi(frame, roi=roi, view=view,
                                                    return_areas = True,
                                                    return_confs = True)
            else:
                pos, ar, con = self.detect_roi(frame, roi=roi, view=view,
                                               return_areas = True,
                                               return_confs = True)
            positions.extend(pos)
            areas.extend(ar)
            confs.extend(con)
        return np.array(positions), np.array(areas), np.array(confs), img

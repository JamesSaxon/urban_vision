import cv2, numpy as np

from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

from PIL import Image

class Detector():

    colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]

    def __init__(self, 
                 model = "../models/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite", 
                 labels = "../models/coco_labels.txt", 
                 relative_coord = False, keep_aspect_ratio = False,
                 categs = ["person"], thresh = 0.6, k = 1,
                 loc = "upper center", verbose = False):

        self.engine = DetectionEngine(model)
        self.labels = dataset_utils.read_label_file(labels) if labels else None

        self.categs = categs

        self.thresh = thresh
        self.k      = k

        self.relative_coord = relative_coord
        self.keep_aspect_ratio = keep_aspect_ratio

        self.roi    = None

        loc = loc.split(" ")
        self.vloc, self.hloc = loc[0], loc[1]

        if self.vloc not in ["upper", "middle", "lower"]:
            raise(ValueError, "Vertical location must be upper, middle, or lower.")

        if self.hloc not in ["left", "center", "right"]:
            raise(ValueError, "Horizontal location must be left, center, or right.")

        self.verbose = verbose


    def detect(self, img, color = None, 
               return_image = False, return_areas = False, return_confs = False):


        tensor = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ans = self.engine.detect_with_image(tensor, 
                                            threshold = self.thresh, top_k = self.k, 
                                            keep_aspect_ratio = False, relative_coord = False)

        XY, AREAS, CONFS = [], [], []
        for obj in ans:
  
            # If the labels file and category filter
            # are both defined, then filter.
            label = None
            if self.labels is not None and len(self.categs):

                label = self.labels[obj.label_id]
                if label not in self.categs: continue


            if self.labels is not None and self.verbose:
                print(self.labels[obj.label_id] + ",", end = " ")
  
            # Draw a rectangle.
            box = obj.bounding_box.flatten()
            xmin, ymin, xmax, ymax = box
            
            if color is None: 
                color = Detector.colors[self.categs.index(label)] if len(self.categs) else (0, 0, 255)

            if return_image: cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)

            if self.verbose: 
                print('conf. = ', obj.score)
                print('-----------------------------------------')

            if self.hloc == "left":   x = xmax
            if self.hloc == "center": x = (xmax + xmin) / 2
            if self.hloc == "right":  x = xmin

            if self.vloc == "upper":  y = ymin
            if self.vloc == "middle": y = (ymax + ymin) / 2
            if self.vloc == "lower":  y = ymax

            XY.append((x, y))
            AREAS.append((xmax - xmin) * (ymax - ymin))
            CONFS.append(obj.score)

        retval = [np.array(XY)]

        if return_areas : retval.append(np.array(AREAS))
        if return_confs : retval.append(np.array(CONFS))
        if return_image : retval.append(img)

        return retval



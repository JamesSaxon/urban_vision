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


    def detect_roi(self, frame, roi=None, view=False, return_areas = False, return_confs = False):
        '''
        Inputs: image, roi coordinates,
        '''
        xmin = roi[0]
        ymin = roi[1]
        xmax = roi[2]
        ymax = roi[3]

        img = frame
        img_detect = frame[ymin:ymax, xmin:xmax]
        image = Image.fromarray(cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB))
        ans = self.engine.detect_with_image(image, threshold = self.thresh, keep_aspect_ratio=False, relative_coord=False, top_k=self.k)
        XY, AREAS, CONFS = [], [], []
        if ans:
            for obj in ans:
                box = obj.bounding_box.flatten()
                box[0] += xmin
                box[2] += xmin
                box[1] += ymin
                box[3] += ymin
                draw_box = box.astype(int)
                if view: cv2.rectangle(img, tuple(draw_box[:2]), tuple(draw_box[2:]), (0,255,255), 2)
                x = int((box[0] + box[2])//2)
                y = int((box[1] + box[3])//2)
                XY.append((x,y))
                AREAS.append((box[2]-box[0])*(box[3]-box[1]))
                CONFS.append(obj.score)
        retval = [XY]
        if return_areas: retval.append(AREAS)
        if return_confs : retval.append(CONFS)
        if view: retval.append(img)

        return retval



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

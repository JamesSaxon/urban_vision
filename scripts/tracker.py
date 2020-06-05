import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import deque

import cv2

from pykalman import KalmanFilter

bgr_colors = [(60, 15, 150), (30, 80, 190), (0, 190, 190), 
              (50, 90, 25), (130, 30, 15), (60, 45, 180)]

class Object():


    kalman_transition = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    kalman_observation = [[1, 0, 0, 0], [0, 0, 1, 0]]


    def __init__(self, id, color = None, kalman_cov = 1):

        self.id = id

        self.xyt  = []
        self.area = []
        self.conf = []
        self.ts   = []
        self.missing = 0

        self.current_area = 0

        if color is None:

            self.color = bgr_colors[self.id % len(bgr_colors)]

        else: 
        
            self.color = color

        self.nobs = 0
        self.active = True

        self.kalman_cov = kalman_cov

    def make_kalman(self):

        initial_state_mean = [self.xyt[0][0], 0, self.xyt[0][1], 0]

        self.kalman = KalmanFilter(transition_matrices = Object.kalman_transition,
                                   observation_matrices = Object.kalman_observation,
                                   initial_state_mean = initial_state_mean,
                                   observation_covariance = np.eye(2) * self.kalman_cov)

        self.kf_means = [self.xyt[0][0], 0, self.xyt[0][1], 0]
        self.kf_covs  = [0, 0, 0, 0]


    def update(self, x, y, t, area = None, conf = None, ts = None):

        # Yay recursion :-)
        if self.xyt and t - 1 != self.last_time:
            self.update(np.nan, np.nan, t - 1)

        self.xyt.append((x, y, t))
        self.area.append(area)
        self.conf.append(conf)
        self.ts.append(ts)

        self.current_area = area

        if len(self.xyt) == 1: 

            self.make_kalman()
            self.nobs += 1

        else:
            
            if x is np.nan:

                self.kf_means, self.kf_covs = self.kalman.filter_update(self.kf_means, self.kf_covs, None)

            else: 

                self.kf_means, self.kf_covs = self.kalman.filter_update(self.kf_means, self.kf_covs, np.array([x, y]))
                self.nobs += 1


    def set_color(self, c = None):

        if c is not None: self.color = c

    @property
    def last_time(self):

        if not(self.xyt): return None

        return self.xyt[-1][2]

    @property
    def last_location(self):

        return self.xyt[-1][:2]

    def predict_location(self, t):

        means, covs = self.kf_means, self.kf_covs

        for t_ in range(self.last_time, t):

            means, covs = self.kalman.filter_update(means, covs, None)
        
        
        return means[0], means[2]



    def kalman_smooth(self, kalman_cov = 0, depth = 0):

        if len(self.xyt) < 2: return self.xyt

        measurements = np.array(self.xyt)[:,:2]
        measurements = np.ma.array(measurements, mask = np.isnan(measurements))

        if depth and depth < len(measurements): 
            measurements = measurements[-depth:,:]
            times = [t for x, y, t in self.xyt[-depth:]]
        else:
            times = [t for x, y, t in self.xyt]


        start_smooth = 0
        while np.isnan(measurements[start_smooth, 0]): start_smooth += 1

        initial_state_mean = [measurements[start_smooth, 0], 0,
                              measurements[start_smooth, 1], 0]

        kf = KalmanFilter(transition_matrices    = Object.kalman_transition,
                          observation_matrices   = Object.kalman_observation,
                          initial_state_mean     = initial_state_mean,
                          observation_covariance = np.eye(2) * kalman_cov)

        means, covariances = kf.smooth(measurements)

        return np.array([means[:,0], means[:,2], times]).T


    def out_of_roi(self, t, roi, roi_buffer = 0):

        if roi is None: return False

        x, y = self.predict_location(t)

        if x < roi["xmin"] + roi_buffer: return True
        if x > roi["xmax"] - roi_buffer: return True
        if y < roi["ymin"] + roi_buffer: return True
        if y > roi["ymax"] - roi_buffer: return True

        return False


    @property
    def df(self):

        df = pd.DataFrame(self.xyt, columns = ["x", "y", "t"])

        df["ts"]   = self.ts
        df["o"]    = self.id

        if any(self.area): df["area"] = self.area
        if any(self.conf): df["conf"] = self.conf

        return df

    def deactivate(self):

        self.active = False



class Tracker():

    def __init__(self, 
                 max_missing = 4, max_distance = 50, 
                 min_distance_overlap = 0.02,
                 predict_match_locations = False,
                 kalman_cov = 0,
                 contrail = 0, color = (255, 255, 255)):

        self.oid = 0
        self.objects = {}

        self.t = 0

        self.predict_match_locations = predict_match_locations
        self.kalman_cov = kalman_cov

        self.CONTRAIL = contrail

        self.MAX_MISSING = max_missing

        self.MAX_DISTANCE    = max_distance
        self.MIN_DISTANCE_OR = min_distance_overlap

        self.color = color

        self.roi = None
        self.roi_buffer = 0

    def set_roi(self, roi, roi_buffer = 0):

        self.roi = roi
        self.roi_buffer = roi_buffer


    def new_object(self):

        self.objects[self.oid] = Object(self.oid)

        self.oid += 1

        return self.oid - 1

    def get_last_locations(self, return_unmatched = False, return_dict = False, min_obs = 0):

        object_unmatched = {k for k, v in self.objects.items() 
                            if v.active and v.nobs > min_obs}

        locations = np.array([self.objects[k].last_location
                              for k in object_unmatched])

        if return_unmatched: return object_unmatched, locations
        if return_dict:      return {o : l for o, l in zip(object_unmatched, locations)}

        return locations


    def predict_current_locations(self, return_unmatched = False, min_obs = 0):

        object_unmatched = {k for k, v in self.objects.items() 
                            if v.active and v.nobs > min_obs}

        locations = np.array([self.objects[k].predict_location(self.t)
                              for k in object_unmatched])

        if return_unmatched: return object_unmatched, locations

        return locations


    def edge_veto(self, box):

        if box.xmin < self.roi["xmin"] + self.roi_buffer: return True
        if box.xmax > self.roi["xmax"] - self.roi_buffer: return True
        if box.ymin < self.roi["ymin"] + self.roi_buffer: return True
        if box.ymax > self.roi["ymax"] - self.roi_buffer: return True

        return False

    def deactivate_objects(self):

        for o in self.objects.values():

            if not o.active: continue

            if self.t - o.last_time > self.MAX_MISSING:
                o.deactivate()
                continue

            if o.out_of_roi(t = self.t, roi = self.roi, roi_buffer = self.roi_buffer):
                o.deactivate()


    def update(self, detections, ts = None):

        self.t += 1

        # If there are no new points, abort.
        if not len(detections):

            self.deactivate_objects()
            return


        # If there are no existing, active points, add them and abort!
        if not self.n_active:
            for idx, det in enumerate(detections):

                if self.edge_veto(det.box): continue

                oidx = self.new_object()
                # self.objects[oidx].set_color(det.color)

                self.objects[oidx].update(det.x, det.y, self.t, det.area, det.conf, ts)

            return

        new_points = np.array([det.xy   for det in detections])
        new_areas  = np.array([det.area for det in detections])
        new_indexes = set(range(len(new_points)))

        if self.predict_match_locations:
            obj_unmatched, obj_points = self.predict_current_locations(return_unmatched = True)

        else:
            obj_unmatched, obj_points = self.get_last_locations(return_unmatched = True)

        obj_indexes = {ki : k for ki, k in enumerate(obj_unmatched)}

        D = cdist(obj_points, new_points)

        D = np.ma.array(D, mask = D > self.MAX_DISTANCE * np.sqrt(new_areas)[np.newaxis,:])
  
        while not D.mask.all():

            # This is the 2D-index
            idx = np.unravel_index(D.argmin(axis = None), D.shape)
        
            # This object and this observation
            #    are now "spoken for."
            D.mask[idx[0],:] = True
            D.mask[:,idx[1]] = True

            # Save the location and current time.
            obj_idx = obj_indexes[idx[0]]
            new_idx = idx[1]

            # new_xy = detections[new_idx].xy
            det = detections[new_idx]

            if self.edge_veto(det.box):
                self.objects[obj_idx].deactivate()
            
            self.objects[obj_idx].update(det.x, det.y, self.t, det.area, det.conf, ts = ts)

            # We won't have to deal with this one.
            new_indexes -= {idx[1]}

        # The new_indexes are not yet new -- just potential/unmatched.
        # If they have a distance within the distance cut-off to an existing object
        # then we don't want to create a new object for them.
        # DO NOT create new objects in the ROI cut-off or buffer.
        D.mask = False
        for idx in new_indexes:

            if D[:,idx].min() < self.MAX_DISTANCE * np.sqrt(new_areas[idx]): continue

            # x, y = new_points[idx]
            det = detections[idx]

            if self.edge_veto(det.box): continue
            
            oidx = self.new_object()

            self.objects[oidx].update(det.x, det.y, self.t, det.area, det.conf, ts = ts)
            # self.objects[oidx].set_color(det.color)


        self.deactivate_objects()


    def draw(self, img, min_obs = 5, scale = 1):

        depth = self.CONTRAIL

        for o in self.objects.values():

            if o.last_time < self.t - depth: continue
            if o.nobs < min_obs: continue

            x0, y0, t0 = None, None, None

            xyt = o.xyt if not self.kalman_cov else o.kalman_smooth(kalman_cov = self.kalman_cov, depth = int(1.5 * depth))

            for x1, y1, t1 in xyt:

                # If not Kalman smoothing, these can be empty.
                if np.isnan(x1): continue

                x1, y1 = int(x1 / scale), int(y1 / scale)

                if depth and self.t - t1 > depth: continue

                if t0 is not None:
                    img = cv2.line(img, tuple([x0, y0]),
                                        tuple([x1, y1]), o.color, 2)

                    img = cv2.circle(img, tuple([x1, y1]), 3, o.color, -1)

                x0, y0, t0 = x1, y1, t1 

            if o.active: 

                if self.predict_match_locations:
                    x, y = o.predict_location(self.t)
                else:
                    x, y = o.last_location

                xy = tuple([int(x / scale), int(y / scale)])
                img = cv2.circle(img, xy, 5, (255, 255, 255), -1)
                img = cv2.circle(img, xy, 5, o.color, 2)

                if o.current_area:
                    length = np.sqrt(o.current_area)
                    img = cv2.circle(img, xy, int(self.MAX_DISTANCE * length / scale), o.color, 1)

        return img

    def write(self, output):

        if self.oid:

            df_out = pd.concat([o.df for o in self.objects.values()])
            df_out.to_csv(output, index = False)

        else:

            pd.DataFrame(columns = ["x", "y", "t", "ts", "o", "area", "conf"]).to_csv(output, index = False)

    @property
    def n_active(self):

        return sum(o.active for o in self.objects.values())





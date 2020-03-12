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

        self.xyt = []
        self.area = []
        self.conf = []
        self.ts   = []
        self.missing = 0

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


        initial_state_mean = [measurements[0, 0], 0, measurements[0, 1], 0]

        kf = KalmanFilter(transition_matrices    = Object.kalman_transition,
                          observation_matrices   = Object.kalman_observation,
                          initial_state_mean     = initial_state_mean,
                          observation_covariance = np.eye(2) * kalman_cov)

        means, covariances = kf.smooth(measurements)

        return np.array([means[:,0], means[:,2], times]).T


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

    def __init__(self, max_missing = 4, max_distance = 50, 
                 predict_match_locations = False, contrail = 0, color = (255, 255, 255)):

        self.oid = 0
        self.objects = {}

        self.t = 0

        self.predict_match_locations = predict_match_locations

        self.CONTRAIL = contrail
        self.MAX_MISSING = max_missing
        self.MAX_DISTANCE = max_distance

        self.color = color


    def new_object(self):

        self.objects[self.oid] = Object(self.oid)

        self.oid += 1

        return self.oid - 1

    def get_last_locations(self, return_unmatched = False, min_obs = 0):

        object_unmatched = {k for k, v in self.objects.items() 
                            if v.active and v.nobs > min_obs}

        locations = np.array([self.objects[k].last_location
                              for k in object_unmatched])

        if return_unmatched: return object_unmatched, locations

        return locations


    def predict_current_locations(self, return_unmatched = False, min_obs = 0):

        object_unmatched = {k for k, v in self.objects.items() 
                            if v.active and v.nobs > min_obs}

        locations = np.array([self.objects[k].predict_location(self.t)
                              for k in object_unmatched])

        if return_unmatched: return object_unmatched, locations

        return locations


    def update(self, new_points, areas = None, confs = None, ts = None, colors = None):

        self.t += 1

        # If there are no new points, abort.
        if new_points is None or not len(new_points):

            # Get rid of objects "hanging around..."
            for o in self.objects.values():
                if self.t - o.last_time > self.MAX_MISSING:
                    o.deactivate()

            return

        if type(colors) is not list: colors = [colors for x in new_points]


        # If there are no existing, active points, add them and abort!
        if not self.n_active:
            for idx, pt in enumerate(new_points):

                oidx = self.new_object()
                self.objects[oidx].set_color(colors[idx])

                a  = None if areas is None else areas[idx]
                c  = None if confs is None else confs[idx]

                self.objects[oidx].update(pt[0], pt[1], self.t, a, c, ts)

            return

        new_points = np.array(new_points)
        new_indexes = set(range(len(new_points)))

        if self.predict_match_locations:
            obj_unmatched, obj_points = self.predict_current_locations(return_unmatched = True)

        else:
            obj_unmatched, obj_points = self.get_last_locations(return_unmatched = True)

        obj_indexes = {ki : k for ki, k in enumerate(obj_unmatched)}

        D = cdist(obj_points, new_points)

        if areas is None:
            D = np.ma.array(D, mask = D > self.MAX_DISTANCE)
        else:
            D = np.ma.array(D, mask = D > self.MAX_DISTANCE * np.sqrt(areas)[np.newaxis,:])
  
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

            new_xy = new_points[new_idx]
            
            if areas is None:
                self.objects[obj_idx].update(new_xy[0], new_xy[1], self.t, ts = ts)

            else:
                self.objects[obj_idx].update(new_xy[0], new_xy[1], self.t, 
                                             areas[new_idx], confs[new_idx], ts = ts)

            # We won't have to deal with this one.
            new_indexes -= {idx[1]}

        # The new_indexes are not yet new -- just potential/unmatched.
        # If they have a distance within the distance cut-off to an existing object
        # then we don't want to create a new object for them.
        D.mask = False
        for idx in new_indexes:

            if areas is None:
                if D[:,idx].min() < self.MAX_DISTANCE: continue

            else: 
                if D[:,idx].min() < self.MAX_DISTANCE * np.sqrt(areas[idx]): continue
            
            oidx = self.new_object()

            if areas is None:
                self.objects[oidx].update(new_points[idx][0], new_points[idx][1], self.t, ts = ts)

            else:
                self.objects[oidx].update(new_points[idx][0], new_points[idx][1], self.t,
                                            areas[idx], confs[idx], ts = ts)

            self.objects[oidx].set_color(colors[idx])


        for o in self.objects.values():
            if self.t - o.last_time > self.MAX_MISSING:
                o.deactivate()


    def draw(self, img, scale = 1, depth = None, kalman_cov = 0):

        if depth is None:
            depth = self.CONTRAIL

        for o in self.objects.values():

            x0, y0, t0 = None, None, None

            xyt = o.xyt if not kalman_cov else o.kalman_smooth(kalman_cov, 2 * depth)

            for x1, y1, t1 in xyt:

                # If not Kalman smoothing, these can be empty.
                if x1 is np.nan: continue

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

        return img

    def write(self, output):

        df_out = pd.concat([o.df for o in self.objects.values()])

        df_out.to_csv(output, index = False)

    @property
    def n_active(self):

        return sum(o.active for o in self.objects.values())





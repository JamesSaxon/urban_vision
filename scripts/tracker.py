import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import deque

import cv2

from pykalman import KalmanFilter

class Object():


    kalman_transition = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    kalman_observation = [[1, 0, 0, 0], [0, 0, 1, 0]]


    def __init__(self, id, kalman_cov = 1):

        self.id = id

        self.xyt = []
        self.missing = 0

        self.color = (0, 0, 255)

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

        


    def update(self, x, y, t):

        # Yay recursion :-)
        if self.xyt and t - 1 != self.last_time:
            self.update(np.nan, np.nan, t - 1)

        self.xyt.append((x, y, t))

        if len(self.xyt) == 1: self.make_kalman()
        else:
            
            if x is np.nan:

                self.kf_means, self.kf_covs = self.kalman.filter_update(self.kf_means, self.kf_covs, None)

            else: 

                self.kf_means, self.kf_covs = self.kalman.filter_update(self.kf_means, self.kf_covs, np.array([x, y]))


    def set_color(self, c):

        self.color = c

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

        initial_state_mean = [self.xyt[0][0], 0, self.xyt[0][1], 0]

        kf = KalmanFilter(transition_matrices = Object.kalman_transition,
                           observation_matrices = Object.kalman_observation,
                           initial_state_mean = initial_state_mean,
                           observation_covariance = np.eye(2) * kalman_cov)

        measurements = np.array(self.xyt)[:,:2]
        measurements = np.ma.array(measurements, mask = np.isnan(measurements))
        if depth: measurements = measurements[-depth,:]

        means, covariances = kf.smooth(measurements)

        return np.array([means[:,0], means[:,2], [t for x, y, t in self.xyt]]).T


    @property
    def df(self):

        df = pd.DataFrame(self.xyt, columns = ["x", "y", "t"])
        df["o"] = self.id

        return df

    def deactivate(self):

        self.active = False



class Tracker():

    def __init__(self, max_missing = 4, max_distance = 50, contrail = 0, color = (255, 255, 255)):

        self.oid = 0
        self.objects = {}

        self.t = 0

        self.CONTRAIL = contrail
        self.MAX_MISSING = max_missing
        self.MAX_DISTANCE = max_distance

        self.color = color


    def new_object(self, x, y, t, c):

        self.objects[self.oid] = Object(self.oid)
        self.objects[self.oid].update(x, y, t)
        self.objects[self.oid].set_color(c)

        self.oid += 1

    def predict_current_locations(self):

        object_unmatched = {k for k, v in self.objects.items() if v.active}

        return np.array([self.objects[k].predict_location(self.t)
                         for k in object_unmatched])

    def update(self, new_points, colors = None):

        self.t += 1

        # If there are no new points, abort.
        if new_points is None or not len(new_points): return

        if colors is None or not len(colors): 
            colors = [self.color for x in new_points]

        # If there are no existing points, add them and abort!
        if not len(self.objects):
            for pt, col in zip(new_points, colors):
                self.new_object(pt[0], pt[1], self.t, col)

            return


        new_points = np.array(new_points)
        new_indexes = set(range(len(new_points)))

        object_unmatched = {k for k, v in self.objects.items() if v.active}
        object_indexes   = {ki : k for ki, k in enumerate(object_unmatched)}
        object_points    = np.array([self.objects[k].predict_location(self.t)
                                     for k in object_unmatched])

        D = cdist(object_points, new_points)
        D = np.ma.array(D, mask = D > self.MAX_DISTANCE)
  
        while not D.mask.all():

            idx = np.unravel_index(D.argmin(axis = None), D.shape)
        
            # This object and this observation
            #    are now "spoken for."
            D.mask[idx[0],:] = True
            D.mask[:,idx[1]] = True

            # Save the location and current time.
            object_idx = object_indexes[idx[0]]

            new_xy = new_points[idx[1]]
            self.objects[object_idx].update(new_xy[0], new_xy[1], self.t)
            self.objects[object_idx].set_color(colors[idx[1]])

            # We won't have to deal with this one.
            new_indexes -= {idx[1]}

        for idx in new_indexes:
            self.new_object(new_points[idx][0], new_points[idx][1], self.t, colors[idx])


        for o in self.objects.values():
            if self.t - o.last_time > self.MAX_MISSING:
                o.deactivate()


    def draw(self, img, scale = 1, color = (0, 0, 255), depth = None, kalman_cov = 0):

        if depth is None:
            depth = self.CONTRAIL

        for o in self.objects.values():

            x0, y0, t0 = None, None, None

            xyt = o.xyt if not kalman_cov else o.kalman_smooth(kalman_cov)

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

        return img

    def write(self, output):

        df_out = pd.concat([o.df for o in self.objects.values()])

        df_out.to_csv(output, index = False)




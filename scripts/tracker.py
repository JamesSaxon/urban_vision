import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from collections import deque
from itertools import permutations, combinations

from detector import Detection, BBox

import cv2

from pykalman import KalmanFilter


bgr_colors = [( 0,  0, 255), ( 0, 155, 255), (0, 255, 255), 
              ( 0, 255, 0),  (255, 0, 0),   (255, 0, 255)]

class Object():
    """
    Objects are sequences of detections that have been associated together, by the Tracker.
    It tracks its locations, confidences, and area over time, as well as its current bounds,
      missing detections, and `edge_strikes` against the bounds of the detection region.
    Objects start out with `active` set to True; this changes to false 
      after a configurable number of missed detections edge strikes.
    The `Object` is also able to project itself forward in time, using a Kalman filter.
    """

    kalman_transition = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    kalman_observation = [[1, 0, 0, 0], [0, 0, 1, 0]]


    def __init__(self, id, color = None, kalman_track_cov = 0):

        self.id = id

        self.xyt  = []
        self.area = []
        self.conf = []
        self.ts   = []

        self.missing = 0
        self.edge_strikes = 0

        self.box = None

        self.current_area = 0

        self.last_detection = None

        if color is not None: self.color = color
        else:

            self.color = bgr_colors[self.id % len(bgr_colors)]


        self.nobs = 0
        self.active = True

        self.kalman_track_cov = kalman_track_cov


    def make_kalman(self):

        initial_state_mean = [self.xyt[0][0], 0, self.xyt[0][1], 0]

        self.kalman = KalmanFilter(transition_matrices = Object.kalman_transition,
                                   observation_matrices = Object.kalman_observation,
                                   initial_state_mean = initial_state_mean,
                                   observation_covariance = np.eye(2) * self.kalman_track_cov)

        self.kf_means = [self.xyt[0][0], 0, self.xyt[0][1], 0]
        self.kf_covs  = [0, 0, 0, 0]


    def fillna(self, t):

        self.xyt.append((np.nan, np.nan, t))
        self.area.append(None)
        self.conf.append(None)
        self.ts.append(None)


    def update(self, det, t, ts = None):
        """
        Update location with either known or missing coordinates (Kalman).
        """

        # Yay recursion :-)
        if self.xyt and t - 1 != self.last_time:
            self.update(None, t - 1)

        if det: 

            self.last_detection = t

            self.xyt.append((det.x, det.y, t))
            self.area.append(det.area)
            self.conf.append(det.conf)
            self.ts.append(ts)

            self.current_area = det.area
            self.box = det.box

        else: self.fillna(t)

        if len(self.xyt) == 1: 

            self.make_kalman()
            self.nobs += 1

        else:
            
            if det is None:

                self.kf_means, self.kf_covs = self.kalman.filter_update(self.kf_means, self.kf_covs, None)

            else: 

                self.kf_means, self.kf_covs = self.kalman.filter_update(self.kf_means, self.kf_covs, np.array(det.xy))
                self.nobs += 1


    def update_track(self, x, y, t, box, ts = None):

        self.xyt.append((x, y, t))
        self.ts.append(ts)

        self.box = box
        self.current_area = box.area

        self.area.append(box.area)
        self.conf.append(np.nan)

        self.kf_means, self.kf_covs = self.kalman.filter_update(self.kf_means, self.kf_covs, np.array((x, y)))


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

        if self.nobs < 4: return self.xyt[-1][:2]

        means, covs = self.kf_means, self.kf_covs

        for t_ in range(self.last_time, t):

            means, covs = self.kalman.filter_update(means, covs, None)
        
        
        return means[0], means[2]



    def kalman_smooth(self, kalman_cov = 0, depth = 0):

        if len(self.xyt) < 2: return self.xyt

        measurements = np.array(self.xyt)[:,:2]
        measurements = np.ma.array(measurements, mask = np.isnan(measurements))

        if depth and depth < len(measurements): 

            while np.ma.is_masked(measurements[-depth, 0]):

                depth -= 1
                if not depth: return np.array([])

            measurements = measurements[-depth:,:]
            times = [t for x, y, t in self.xyt[-depth:]]

        else: times = [t for x, y, t in self.xyt]


        initial_state_mean = [measurements[0, 0], 0,
                              measurements[0, 1], 0]

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

    def deactivate(self, t):

        self.active = False
        self.t_deactive = t

        self.tracker = None



class Tracker():
    """
    Tracker associates subsequent detections together.
    It can also run CSRT correlational tracking, 
      but this is expensive and doesn't work very well.
    """

    STRIKES = 1

    def __init__(self, 
                 method = "min_cost",
                 max_missing = 4, max_distance = 50, 
                 min_distance_overlap = 0.02,
                 max_track = 0, candidate_obs = 0,
                 predict_match_locations = False,
                 kalman_track_cov = 0, 
                 kalman_viz_cov = 0, 
                 roi_loc = "upper center",
                 contrail = 0, color = (255, 255, 255)):

        self.oid = 0
        self.objects = {}

        self.t = 0

        self.predict_match_locations = predict_match_locations
        self.kalman_track_cov = kalman_track_cov
        self.kalman_viz_cov = kalman_viz_cov

        self.CONTRAIL = contrail

        self.MAX_MISSING = max_missing

        self.MAX_DISTANCE     = max_distance
        self.MAX_DISTANCE2    = max_distance**2
        self.MIN_DISTANCE_OR  = min_distance_overlap
        self.MIN_DISTANCE_OR2 = min_distance_overlap**2

        self.NOBS_CANDIDATE = candidate_obs

        self.INCLUDE_CANDIDATES_CYCLES = [True]
        if self.NOBS_CANDIDATE: 
            self.INCLUDE_CANDIDATES_CYCLES = [False, True]

        self.MAX_TRACK = max_track

        self.color = color

        self.roi = None
        self.roi_buffer = 0

        if method in ["greedy", "min_cost"]:
            self.method = method
        else: 
            print("Match method must be either 'greedy' or 'min_cost'.  {} is neither!\nSetting to min_cost.")
            self.method = "min_cost"


        loc = roi_loc.split(" ")
        self.vloc, self.hloc = loc[0], loc[1]

        if self.vloc not in ["upper", "middle", "lower"]:
            raise(ValueError, "Vertical location must be upper, middle, or lower.")

        if self.hloc not in ["left", "center", "right"]:
            raise(ValueError, "Horizontal location must be left, center, or right.")


    def set_roi(self, roi, roi_buffer = 0):

        self.roi = roi
        self.roi_buffer = roi_buffer


    def new_object(self):

        self.objects[self.oid] = Object(self.oid)

        self.oid += 1

        return self.oid - 1

    def get_last_locations(self, return_unmatched = False, return_dict = False):

        object_unmatched = np.array([k for k, v in self.objects.items() if v.active])

        locations = np.array([self.objects[k].last_location
                              for k in object_unmatched])

        if return_unmatched: return object_unmatched, locations
        if return_dict:      return {o : l for o, l in zip(object_unmatched, locations)}

        return locations


    def predict_current_locations(self, return_unmatched = False):

        object_unmatched = np.array([k for k, v in self.objects.items() if v.active])

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
        """
        Get rid of objects that have struck the edge too many times, or have left the ROI.
        """

        for o in self.objects.values():

            if not o.active: continue

            if self.edge_veto(o.box): o.edge_strikes += 1

            if self.t - o.last_detection > self.MAX_MISSING or \
               o.out_of_roi(t = self.t, roi = self.roi, roi_buffer = self.roi_buffer) or \
               o.edge_strikes > Tracker.STRIKES: 
                o.deactivate(self.t)


    def match(self, D2, mask_candidates): 
        """
        Match objects between two frames based on and m x n matrix of squared distances.
        There are two possible ways of doing this -- one greed, the other evaluating all combinatorics
          for the least total cost.
        The cut-off, MAX_DISTANCE2 has already been applied to create the mask,
          before passing the distance matrix, D2, to this function.
        """

        if   self.method == "greedy":    return self.greedy_matches  (D2, mask_candidates)
        elif self.method == "min_cost" : return self.min_cost_matches(D2, mask_candidates)

    def greedy_matches(self, D2, mask_candidates):
        """
        This method simply takes the shortest distance between 
          objects in the present and last frames.
        After each call of argmin, that object (in each frame) is removed from consideration -- 
          the row and column are masked off.
        The function terminates when there are no available objects combination to match.
        """

        matches = {}

        # This is just going False / True.
        # We run the cycle once with candidates excluded, and then include them.
        for include_candidates in self.INCLUDE_CANDIDATES_CYCLES:

            while (    include_candidates and not (D2.mask).all()) or \
                  (not include_candidates and not (D2.mask | mask_candidates).all()):

                # This is the 2D2-index
                if not include_candidates: # Preferentially mask the existing objects.
                    idx = np.unravel_index(np.ma.array(D2, mask = np.logical_or(D2.mask, mask_candidates))\
                                                        .argmin(axis = None), D2.shape)

                else: # Only THEN consider new ones.
                    idx = np.unravel_index(D2.argmin(axis = None), D2.shape)
            
                # This object and this observation 
                #     are now "spoken for."
                D2.mask[idx[0],:] = True # Old
                D2.mask[:,idx[1]] = True # New

                # Save for return...
                matches[idx[0]] = idx[1] # Old -> New

        return matches

    def min_cost_matches(self, D2, mask_candidates = None):
        """
        Instead of greedy matching, we can also consider all possible permutations of matches.
        For this, we evaluate the total number matched, and total cost of the match D2.sum().
        The "optimal" choice is the one with the lowest cost, 
          among choices with the maximum number of matches.
        This is somewhat slower, but the permutations are not too bad,
          since we are considering objects currently in view.
        """
        
        full_matches = {}

        # As in the greedy match, 
        #  first run it with candidates masked, and then include them.
        extra_masks = [False]
        if mask_candidates is not None:
            extra_masks = [mask_candidates, False]

        for aux_mask in extra_masks:

            nmatches  = 0
            min_dist2 = np.inf

            # Lots of negatives here -- lists of elements that are not all masked 
            #    ("all" of masks is false).
            old_objs = np.argwhere((D2.mask | aux_mask).all(axis = 1) == False).flatten()
            new_objs = np.argwhere(D2.mask.all(axis = 0) == False).flatten()

            # Get out of here if either is empty!)
            size = min(len(old_objs), len(new_objs))
            if not size: continue

            best_old, best_new = None, None
            
            # ONE of these should be combinations and the 
            #  other should be permutations.
            for old_idx in combinations(old_objs, size):
                for new_idx in permutations(new_objs, size):
                                
                    # Careful: if size is larger than the actually-matchable, 
                    #   then don't match the masked ones!!
                    matches = (D2[old_idx, new_idx].mask == False).sum()
                    dist2   = D2[old_idx, new_idx].sum()
                    
                    # Search for the *most* matches possible, 
                    #   and *within* that category, the lowest distance.
                    if matches > nmatches or \
                       (matches == nmatches and dist2 < min_dist2):
                            
                        min_dist2 = dist2
                        nmatches = matches
                        
                        best_old = old_idx
                        best_new = new_idx
    
            # Mask off the ones we found....
            # possibly another pass coming.
            for old_idx in best_old:
                for new_idx in best_new:

                    # But not if this was masked!!
                    if D2.mask[old_idx,new_idx]: 
                        continue
                    
                    D2.mask[old_idx,:] = True
                    D2.mask[:,new_idx] = True

                    # Have to extend, since we have two passes.
                    full_matches[old_idx] = new_idx

        D2.mask = False
                    
        return full_matches


    def update(self, detections, frame = None, ts = None):
        """
        This is the main function, in which we add new detections
          and associate them with existing ones!
        """

        self.t += 1

        # If there are no new points, abort.
        if not len(detections):

            self.deactivate_objects()
            self.track(frame)
            return


        # If there are no existing, active points, add them and abort!
        if not self.n_active:
            for idx, det in enumerate(detections):

                if self.edge_veto(det.box): continue

                oidx = self.new_object()

                self.objects[oidx].update(det, self.t, ts)

            return


        new_points = np.array([det.xy   for det in detections])
        new_areas  = np.array([det.area for det in detections])
        new_indexes = set(range(len(new_points)))

        # By whichever means, get the current locations of known objects.
        if self.predict_match_locations:
            obj_unmatched, obj_points = self.predict_current_locations(return_unmatched = True)

        else:
            obj_unmatched, obj_points = self.get_last_locations(return_unmatched = True)

        obj_areas = np.array([self.objects[o].current_area for o in obj_unmatched])

        # This is the basis for all matching, whether greedy or minimum total cost.
        D2 = cdist(obj_points, new_points, 'sqeuclidean')

        # The initial mask is based on the squared distance between objects,
        # relative to the sizes / areas of those objects.
        mask_new  = D2 > self.MAX_DISTANCE2 * new_areas[np.newaxis,:]
        mask_old  = D2 > self.MAX_DISTANCE2 * obj_areas[:,np.newaxis]

        mask = mask_new & mask_old

        D2 = np.ma.array(D2, mask = mask) # D2 > self.MAX_DISTANCE * np.sqrt(new_areas)[np.newaxis,:])

        # If this setting is in place, do not match objects
        # until they have been around for a little while.
        # THis mask is passed to the self.match(), 
        #   and applied there (and then not applied, in turn).
        mask_candidates = None
        if self.NOBS_CANDIDATE:
            mask_candidates = np.array([self.objects[o].nobs < self.NOBS_CANDIDATE for o in obj_unmatched])[:,np.newaxis]

        # Do the matching!
        matches = self.match(D2, mask_candidates)
  
        new_indexes -= set(matches.values())
        for old_idx, new_idx in matches.items():

            # Get the object indexes from the array.
            obj_idx = obj_unmatched[old_idx]

            # Grab the detection and update it.
            det = detections[new_idx]
            self.objects[obj_idx].update(det, self.t, ts = ts)

            # And remove if necessary.
            ##  if self.edge_veto(det.box):
            ##      self.objects[obj_idx].deactivate(self.t)
            # Although ... deactivate_objects will do this for the point.
            # and we won't *create* new objects in the veto region...
            

        # The new_indexes are not yet new -- just potential/unmatched.
        # If they have a distance within the distance cut-off to an existing object
        # then we don't want to create a new object for them.
        # DO NOT create new objects in the ROI cut-off or buffer.
        D2.mask = False
        for idx in new_indexes:

            if D2[:,idx].min() < self.MIN_DISTANCE_OR2 * new_areas[idx]: continue

            det = detections[idx]

            if self.edge_veto(det.box): continue
            
            oidx = self.new_object()

            self.objects[oidx].update(det, self.t, ts = ts)


        # Deactivate objects multiple edge strikes,
        # or which have one out of the detection ROI.
        self.deactivate_objects()

        # Correlational tracking (only if MAX_TRACK > 0).
        self.track(frame)


    def reset_track(self, frame = None):
        """
        Create a CSRT tracker 
        """

        for tr_idx, oidx in enumerate(self.objects):

            if not self.objects[oidx].active: continue

            # Only update the tracker if we have a detection.
            if self.objects[oidx].last_detection != self.t:
                # If it has been longer than max_track...
                if self.t - self.objects[oidx].last_detection >= self.MAX_TRACK:
                    self.objects[oidx].tracker = None 
                continue

            box = self.objects[oidx].box
            if box.xmin < 0 or box.ymin < 0: continue
            if box.xmax > frame.shape[1]:    continue
            if box.ymax > frame.shape[0]:    continue

            self.objects[oidx].tracker = cv2.TrackerCSRT_create()
            self.objects[oidx].tracker.init(frame, box.min_and_width())


    def track(self, frame = None, ts = None):

        # Return immediately if MAX_TRACK is 0.
        if not self.MAX_TRACK: return

        # Can't do anything without a frame!!
        if frame is None: return

        for oidx, o in self.objects.items():

            # If it was detected this round, get out -- let the real detection stand!
            if o.last_detection == self.t: continue

            # If the tracker has expired, then also get out!!
            if o.tracker is None: continue

            # If the tracker generates a good return value,
            #  then save that as the new location 
            ret, new_box = o.tracker.update(frame)

            if not ret:
                o.tracker = None
                continue

            # Return value was OK, so make a box out of it!
            new_box = BBox(xmin = new_box[0], ymin = new_box[1], 
                           xmax = new_box[0] + new_box[2], 
                           ymax = new_box[1] + new_box[3])

            x, y = new_box.loc(self.hloc, self.vloc)

            # This is "tracking" based update instead of "detection"-based.
            # This is distinguished by no change in the number of observations
            #  or the last detected frame.
            self.objects[oidx].update_track(x, y, self.t, new_box, ts = ts)

        self.reset_track(frame)


    def draw(self, img, scale = 1):

        depth = self.CONTRAIL

        for o in self.objects.values():

            if o.last_detection < self.t - depth: continue
            # If active, print expected or current location, and rectangles.

            width = 0
            if   self.t == o.last_detection and o.nobs > self.NOBS_CANDIDATE: width = 2
            elif self.t == o.last_detection and o.active: width = 1
            elif (self.t - o.last_time <= self.MAX_TRACK) and o.nobs > self.NOBS_CANDIDATE: width = 1

            if width: img = o.box.draw_rectangle(img, scale = scale, color = o.color, width = width)

            if o.nobs < self.NOBS_CANDIDATE: continue

            x0, y0, t0 = None, None, None

            xyt = o.xyt if not self.kalman_viz_cov \
                        else o.kalman_smooth(kalman_cov = self.kalman_viz_cov, depth = int(1.5 * depth))

            for x1, y1, t1 in xyt:

                # If not Kalman smoothing, these can be empty.
                if np.isnan(x1): continue

                if not o.active and t1 > o.t_deactive: break

                x1, y1 = int(x1 / scale), int(y1 / scale)

                if depth and self.t - t1 > depth: continue

                if t0 is not None:
                    img = cv2.line(img, tuple([x0, y0]),
                                        tuple([x1, y1]), o.color, 2)

                    img = cv2.circle(img, tuple([x1, y1]), 3, o.color, -1)

                x0, y0, t0 = x1, y1, t1 


            if o.active: # and (self.t == o.last_detection or o.nobs >= self.NOBS_CANDIDATE):

                if self.predict_match_locations:
                    x, y = o.predict_location(self.t)
                else:
                    x, y = o.last_location

                xy = tuple([int(x / scale), int(y / scale)])
                # cv2.putText(img, str(o.id), tuple([xy[0]+3, xy[1]-3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                img = cv2.circle(img, xy, 5, (255, 255, 255), -1)
                img = cv2.circle(img, xy, 5, o.color, 2)

                if o.current_area:
                    length = np.sqrt(o.current_area)
                    img = cv2.circle(img, xy, int(self.MIN_DISTANCE_OR * length / scale), (255, 255, 255), 1)
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





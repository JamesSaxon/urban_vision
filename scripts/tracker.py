import numpy as np
from scipy.spatial.distance import cdist
from collections import deque

import cv2

class Object():

    def __init__(self, id):

        self.id = id

        self.xyt = []
        self.missing = 0

        self.color = (0, 0, 255)

        self.active = True


    def update(self, x, y, t):

        self.xyt.append((x, y, t))

    def set_color(self, c):

        self.color = c


    def last_time(self):

        return self.xyt[-1][2]


    def deactivate(self):

        self.active = False


class Tracker():

    def __init__(self, max_missing = 4, max_distance = 50, contrail = 1000, color = (255, 255, 255)):

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

    def update(self, new_points, colors = None):

        self.t += 1

        # If there are no new points, abort.
        if new_points is None: return

        if colors is None: [self.color for x in new_points]

        # If there are no existing points, add them and abort!
        if not len(self.objects):
            for pt, col in zip(new_points, colors):
                self.new_object(pt[0], pt[1], self.t, col)

            return


        new_points = np.array(new_points)
        new_indexes = set(range(len(new_points)))

        object_unmatched = {k for k, v in self.objects.items() if v.active}
        object_indexes   = {ki : k for ki, k in enumerate(object_unmatched)}
        object_points    = np.array([self.objects[k].xyt[-1] for k in object_unmatched])

        
        ## print("Objects - ")
        ## print(object_unmatched)
        ## print(object_indexes)
        ## print(object_points.shape)
        ## print(object_points[:,:2])
        D = cdist(object_points[:,:2], new_points)
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
            if self.t - o.last_time() > self.MAX_MISSING:
                o.deactivate()


    def draw(self, img, color = (0, 0, 255), depth = None):

        if depth is None:
            depth = self.CONTRAIL

        for o in self.objects.values():

            x0, y0, t0 = None, None, None

            for x1, y1, t1 in o.xyt:

                if self.t - t1 > depth: continue

                if t0 is not None:
                    img = cv2.line(img, tuple([x0, y0]),
                                        tuple([x1, y1]), o.color, 2)

                x0, y0, t0 = x1, y1, t1 

        return img




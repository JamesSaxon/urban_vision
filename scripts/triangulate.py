import cv2, numpy as np

import matplotlib.pyplot as plt

from itertools import islice
from collections import deque

def map_format(ax, on = False):

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    if not on:
        ax.set_axis_off()
        ax.set_axis_on()
        for a in ["bottom", "top", "right", "left"]:
            ax.spines[a].set_linewidth(0)

    return ax


class Triangulate():

    def __init__(self, cam1, cam2, calib = "/media/jsaxon/brobdingnag/data/cv/calib/", 
                 xmin = -3, xmax = 3, ymin = -3, ymax = 3):

        self.cam1 = cam1
        self.cam2 = cam2
        
        self.K1 = np.load("{}/{}/K.npy".format(calib, cam1))
        self.K2 = np.load("{}/{}/K.npy".format(calib, cam2))
        
        self.dist1 = np.load("{}/{}/dist.npy".format(calib, cam1))
        self.dist2 = np.load("{}/{}/dist.npy".format(calib, cam2))
        
        self.rvec1 = np.load("{}/{}/rvec.npy".format(calib, cam1))
        self.rvec2 = np.load("{}/{}/rvec.npy".format(calib, cam2))
        
        self.tvec1 = np.load("{}/{}/tvec.npy".format(calib, cam1))
        self.tvec2 = np.load("{}/{}/tvec.npy".format(calib, cam2))
        
        self.P1 = np.dot(self.K1, np.concatenate((cv2.Rodrigues(self.rvec1)[0], self.tvec1), axis=1))
        self.P2 = np.dot(self.K2, np.concatenate((cv2.Rodrigues(self.rvec2)[0], self.tvec2), axis=1))
        
        # Plotting
        fig, ax = plt.subplots()
        
        fig.patch.set_facecolor("k")
        ax.set_facecolor("k")
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        
        map_format(ax)

        self.ax = ax
        
        self.points_3d = []

        self.artists = deque([])

        self.nframe = 0


    def triangulate(self, pts1, pts2, max_reproj_error = 50, plot = True):

        self.points_3d.clear()


        # pts1_norm = cv2.undistortPoints(pts1, cameraMatrix = K1, distCoeffs = dist1)
        # pts2_norm = cv2.undistortPoints(pts2, cameraMatrix = K2, distCoeffs = dist2)
        # points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)

        if pts1 is not None and pts2 is not None: 

            for x1 in pts1:
                for x2 in pts2:

                    points_4d = cv2.triangulatePoints(self.P1, self.P2, np.array([x1]).T, np.array([x2]).T)
                    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1,3)
            
                    # Reprojection error....
                    reproj1, _ = cv2.projectPoints(points_3d, self.rvec1, self.tvec1, self.K1, self.dist1)
                    print(np.sqrt(np.sum((reproj1 - x1)**2)))
                    if np.sqrt(np.sum((reproj1 - x1)**2)) < max_reproj_error:
                        self.points_3d.append(points_3d)
                    
                        print("  .  ", end = "", flush = True)


    def plot(self):

        if len(self.points_3d): 
        
            plot_points = np.array(self.points_3d).reshape(-1, 3)

            a = self.ax.scatter(plot_points[:,0], plot_points[:,1], alpha = 1,
                                s = 1, ec = "w", marker = "o", lw = 0.5, facecolors='none')

            self.artists.appendleft(a)



    def update(self, dt, R = 200, T = 5):

        if not len(self.artists): return

        while len(self.artists) and self.artists[-1].get_alpha() - dt / T < 0:

            self.artists.pop().remove()


        for ai, a in enumerate(self.artists):

            frac = np.sqrt(a.get_alpha())
            frac = frac - dt / T

            alpha = frac ** 2
            size  = [(R * (1 - frac)) ** 2 for g in a.get_sizes()]

            a.set_alpha(alpha)
            a.set_sizes(size)

        plt.pause(0.01)

    def save(self, fname = "/media/jsaxon/brobdingnag/data/cv/vid/rain/{:04}.png"):

        self.ax.figure.savefig(fname.format(self.nframe),
                               bbox_inches='tight', pad_inches=0.05,
                               facecolor = "k", edgecolor='none')
        self.nframe += 1




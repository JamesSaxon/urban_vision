import cv2, numpy as np

import matplotlib.pyplot as plt

from itertools import islice
from collections import deque

from scipy.ndimage import gaussian_filter

import datetime

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
                 depth = 1, xmin = -3, xmax = 3, ymin = -4, ymax = 4, bin_scale = 12, 
                 zmin = -np.inf, zmax = np.inf, zclip = False):

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

        self.P1 = np.concatenate((cv2.Rodrigues(self.rvec1)[0], self.tvec1), axis=1)
        self.P2 = np.concatenate((cv2.Rodrigues(self.rvec2)[0], self.tvec2), axis=1)

        self.zmin = zmin
        self.zmax = zmax
        self.zclip = zclip

        self.points_3d = []
        
        # Plotting
        self.nframe = 0

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

        fig, ax = plt.subplots(figsize = ((xmax - xmin) / 2, (ymax - ymin)/2))

        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        fig.patch.set_facecolor("k")
        ax.set_facecolor("k")
        map_format(ax)

        self.ax = ax

        self.depth = depth
        self.artists = deque([])
        
        self.extent = (xmin, xmax, ymin, ymax)
        self.bin_scale = bin_scale

        self.H = np.zeros((bin_scale * (xmax - xmin), 
                           bin_scale * (ymax - ymin)))
        
        self.view = False

        self.im_hist = None

    def set_view(self, view = None):

        if view is None: self.view = True
        else: self.view = view

    def set_save(self, write, fname = "/media/jsaxon/brobdingnag/data/cv/vid/rain/"):

        if write is None: self.write = True
        else: self.write = write

        self.fname = fname
        self.t = None

        with open(fname + "rain.ffconcat", "w") as out:
            out.write("ffconcat version 1.0\n")


    def triangulate(self, pts1, pts2, max_reproj_error = 50, plot = True, zmin = -np.inf, zmax = np.inf):

        self.points_3d.clear()

        if pts1 is None or pts1.size == 0: return
        if pts2 is None or pts2.size == 0: return

        print(" === ", len(pts1), len(pts2))

        pts1_norm = cv2.undistortPoints(pts1, cameraMatrix = self.K1, distCoeffs = self.dist1)
        pts2_norm = cv2.undistortPoints(pts2, cameraMatrix = self.K2, distCoeffs = self.dist2)

        for x1_raw, x1_calib in zip(pts1, pts1_norm):
            for x2_raw, x2_calib in zip(pts2, pts2_norm):

                points_4d = cv2.triangulatePoints(self.P1, self.P2, np.array([x1_calib]).T, np.array([x2_calib]).T)
                points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1,3)
        
                # Reprojection error....
                reproj1_dist, _ = cv2.projectPoints(points_3d, self.rvec1, self.tvec1, self.K1, self.dist1)
                reproj1_err = np.sqrt(np.sum((reproj1_dist - x1_raw)**2))

                reproj2_dist, _ = cv2.projectPoints(points_3d, self.rvec2, self.tvec2, self.K2, self.dist2)
                reproj2_err = np.sqrt(np.sum((reproj2_dist - x2_raw)**2))

                print(np.round(reproj1_err, 1), np.round(reproj2_err, 1), points_3d, end = " ")

                if np.isnan(reproj1_err) or reproj1_err > max_reproj_error:
                    print()
                    continue

                if np.isnan(reproj2_err) or reproj2_err > max_reproj_error:
                    print()
                    continue

                if self.zclip and (points_3d[0][2] < self.zmin or 
                                   points_3d[0][2] > self.zmax): 
                    print()
                    continue

                print("✔")

                self.points_3d.append(points_3d)
                self.incr_hist(*points_3d.ravel()[:2])


        self.draw_points()


    def incr_hist(self, x, y):

        if x < self.xmin or x > self.xmax or \
           y < self.ymin or y > self.ymax:
           
           return

        bx = int((x - self.xmin) * self.bin_scale)
        by = int((y - self.ymin) * self.bin_scale)

        self.H[bx,by] += 1

    def draw_points(self):

        if not len(self.points_3d): return
        
        plot_points = np.array(self.points_3d).reshape(-1, 3)

        a = self.ax.scatter(plot_points[:,0], plot_points[:,1], alpha = 1, zorder = 5, 
                            s = 20, ec = "w", marker = "o", lw = 1, facecolors = "w")

        self.artists.appendleft(a)


    def update(self, dt = 0, R = 200, T = 5):

        if not len(self.artists): return

        while (self.depth and len(self.artists) > self.depth) or \
              (len(self.artists) and self.artists[-1].get_alpha() - dt / T < 0):

            self.artists.pop().remove()

        if dt > 0:

            for ai, a in enumerate(self.artists):

                frac = np.sqrt(a.get_alpha())
                frac = frac - dt / T

                alpha = frac ** 2

                size  = (R * (1 - frac)) ** 2
                sizes = [size for g in a.get_sizes()]

                face  = "none" if ai or size > 500 else "w"
                faces = [face for g in a.get_sizes()]

                a.set_alpha(alpha)
                a.set_sizes(sizes)
                a.set_facecolors(face)

        if not (self.nframe % 100) and self.nframe > 0:

            if self.im_hist is not None: self.im_hist.remove()

            H = gaussian_filter(self.H.T, 0.25 * self.bin_scale, mode = "constant", cval = 0)

            self.im_hist = self.ax.imshow(H, cmap = "magma", zorder = -5, origin = "lower", extent = self.extent)

            print("Histogram updated.")


        if self.view:  plt.pause(0.0001)
        if self.write: self.save_image()

        self.nframe += 1


    def save_image(self):

        if self.t is None:
            
            self.t = datetime.datetime.now()

        tnow = datetime.datetime.now()
        dt = (tnow - self.t).total_seconds()

        self.t = tnow

        ofile = self.fname + "/{:05d}.png".format(self.nframe)

        self.ax.figure.savefig(ofile, dpi = 50,
                               bbox_inches='tight', pad_inches=0,
                               facecolor = "k", edgecolor='none')

        with open(self.fname + "/rain.ffconcat", "a") as out:
            out.write("file {:05d}.png\n".format(self.nframe))
            out.write("duration {:.05f}\n".format(dt))




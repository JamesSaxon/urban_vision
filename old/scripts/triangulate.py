import cv2, numpy as np, pandas as pd

import matplotlib.pyplot as plt

from itertools import islice
from collections import deque

from scipy.ndimage import gaussian_filter

import statsmodels.formula.api as smf

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

    def __init__(self, cam1, cam2, tracker1, tracker2,
                 calib = "/media/jsaxon/brobdingnag/data/cv/calib/", 
                 xmin = -3, xmax = 3, ymin = -4, ymax = 4,
                 zmin = -10, zmax = 10, zclip = False,
                 bin_scale = 12, depth = 1, verbose = False):

        self.cam1 = cam1
        self.cam2 = cam2

        self.tracker1 = tracker1
        self.tracker2 = tracker2

        self.verbose = verbose
        
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

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax

        self.zclip = zclip
        
        self.xy_noise_scale = 5
        self.create_map(N = 5000)

        self.points_3d = []
        
        # Plotting
        self.nframe = 0

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

    def create_map(self, N = 5000):

        x = self.xmin + (self.xmax - self.xmin) * np.random.rand(N)
        y = self.ymin + (self.ymax - self.ymin) * np.random.rand(N)
        z = self.zmin + (self.zmax - self.zmin) * np.random.rand(N)

        df  = pd.DataFrame({"x" : x, "y" : y, "z" : z})
        pts = df[["x", "y", "z"]].values.reshape(-1,3)
	
        axis_proj1, _ = cv2.projectPoints(pts, self.rvec1, self.tvec1, self.K1, self.dist1)
        axis_proj2, _ = cv2.projectPoints(pts, self.rvec2, self.tvec2, self.K2, self.dist2)
	
        for xi, x in enumerate("xy"):
            df[x + "1"] = axis_proj1[:,0,xi]
            df[x + "2"] = axis_proj2[:,0,xi]

        smf_x = smf.ols("x1 ~ x2 + y2", data = df).fit()
        smf_y = smf.ols("y1 ~ x2 + y2", data = df).fit()

        xmap_res = np.array(list(smf_x.params.to_dict().values()))
        ymap_res = np.array(list(smf_y.params.to_dict().values()))

        self.xymap = np.vstack([xmap_res, ymap_res])

        self.xmap_lim = list(smf_x.resid.quantile([0.001, 0.999]))
        self.ymap_lim = list(smf_y.resid.quantile([0.001, 0.999]))

        ##  df["x1_r"] = smf_x.resid
        ##  df["y1_r"] = smf_y.resid

        ##  df.to_csv("resid.csv", index = False)

        ##  print(self.xmap_lim)
        ##  print(self.ymap_lim)


    def map_2_to_1(self, pts2):

        pts2_arr = np.array(list(pts2.values()))
        pts2_arr = np.append(np.ones((pts2_arr.shape[0], 1)), pts2_arr, axis = 1)

        pts2_c1 = np.matmul(self.xymap, pts2_arr.T).T

        return {o : l for o, l in zip(pts2.keys(), pts2_c1)}


    def associate_trackers(self, pts1, pts2, tol = 50):
	
        pts2_c1 = self.map_2_to_1(pts2)

        df1 = pd.DataFrame.from_dict(pts1,    columns = ["x", "y"], orient='index')
        df2 = pd.DataFrame.from_dict(pts2_c1, columns = ["x", "y"], orient='index')

        df1["idx"] = df1.index
        df1["all"] = 1

        df2["idx"] = df2.index
        df2["all"] = 1

        df = pd.merge(df1, df2, on = "all", suffixes = ("1", "2"))

        df["dx"] = df.x1 - df.x2
        df["dy"] = df.y1 - df.y2

        df["keep_x"] = (self.xmap_lim[0] - tol < df["dx"]) & (df["dx"] < self.xmap_lim[1] + tol)
        df["keep_y"] = (self.ymap_lim[0] - tol < df["dy"]) & (df["dy"] < self.ymap_lim[1] + tol)

        df["dtot2"]  = (self.xy_noise_scale * df["dx"]) ** 2 + df["dy"] ** 2

        df = df[df.keep_x & df.keep_y]\
               .sort_values(by = ["idx1", "dtot2"])\
               .drop_duplicates("idx1", keep = "first")\
               .sort_values(by = ["idx2", "dtot2"])\
               .drop_duplicates("idx2", keep = "first")

        # print(df)

        return df.set_index("idx1").idx2.to_dict()


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


    def triangulate(self, min_obs = 5, max_reproj_error = 100, plot = True):

        self.points_3d.clear()

        pts1 = self.tracker1.get_last_locations(min_obs = min_obs, return_dict = True)
        pts2 = self.tracker2.get_last_locations(min_obs = min_obs, return_dict = True)

        if pts1 is None or not len(pts1): return
        if pts2 is None or not len(pts2): return

        object_match = self.associate_trackers(pts1, pts2)

        if self.verbose: print(" === ", len(pts1), len(pts2), object_match)

        for x1_loc, x2_loc in object_match.items():

            x1_raw = pts1[x1_loc].reshape(1, 1, 2)
            x1_calib = cv2.undistortPoints(x1_raw, cameraMatrix = self.K1, distCoeffs = self.dist1)

            x2_raw = pts2[x2_loc].reshape(1, 1, 2)
            x2_calib = cv2.undistortPoints(x2_raw, cameraMatrix = self.K2, distCoeffs = self.dist2)

            points_4d = cv2.triangulatePoints(self.P1, self.P2, x1_calib.T, x2_calib.T)
            points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1,3)
        
            # Reprojection error....
            reproj1_dist, _ = cv2.projectPoints(points_3d, self.rvec1, self.tvec1, self.K1, self.dist1)
            reproj1_err = np.sqrt(np.sum((reproj1_dist - x1_raw)**2))

            reproj2_dist, _ = cv2.projectPoints(points_3d, self.rvec2, self.tvec2, self.K2, self.dist2)
            reproj2_err = np.sqrt(np.sum((reproj2_dist - x2_raw)**2))

            if self.verbose: print(np.round(reproj1_err, 1), np.round(reproj2_err, 1), points_3d, end = " ")

            if np.isnan(reproj1_err) or reproj1_err > max_reproj_error:
                if self.verbose: print()
                continue

            if np.isnan(reproj2_err) or reproj2_err > max_reproj_error:
                if self.verbose: print()
                continue

            if self.zclip and (points_3d[0][2] < self.zmin or 
                               points_3d[0][2] > self.zmax): 
                if self.verbose: print()
                continue

            if self.verbose or not self.view: print("✔", end = " ")

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

            if self.verbose: print("Histogram updated.")

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




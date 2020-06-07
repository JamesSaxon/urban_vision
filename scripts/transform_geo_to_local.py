#!/Users/jsaxon/anaconda/envs/py-geo/bin/python


import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('Agg')

from fiona.crs import from_epsg
from shapely.geometry import Point

import argparse


def transform_local(geo_dir, tag, angle, epsg):

    df = pd.read_csv("{}/{}.csv".format(geo_dir, tag))
    
    geometry = gpd.GeoSeries([Point(xy) for xy in zip(df.lat, df.lon)], crs = from_epsg(4326))
    
    gdf = gpd.GeoDataFrame(data = df, geometry = geometry)
    gdf.to_crs(epsg = epsg, inplace = True)
    gdf["idx"] = gdf.index
    
    gdf.set_geometry(gdf.geometry.rotate(angle, origin = (0, 0, 0)), inplace = True)
    
    bbox = gdf.unary_union.bounds[:2]
    gdf.set_geometry(gdf.geometry.translate(xoff = -bbox[0], yoff = -bbox[1]), inplace = True)

    gdf["x"] = gdf.geometry.apply(lambda x: x.coords[0][0])
    gdf["y"] = gdf.geometry.apply(lambda x: x.coords[0][1])

    # Just for diagnostics
    bbox = gdf.unary_union.bounds
    ax = gdf.plot(column = "idx", figsize = (2, 2 * (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])))
    ax.figure.savefig("{}/{}_local.pdf".format(geo_dir, tag),
                      bbox_inches = "tight", pad_inches = 0.1)
    
    gdf = gdf[["id", "lon", "lat", "x", "y"]].copy()
    gdf.to_csv("{}/{}_local.csv".format(geo_dir, tag), 
               index = False, float_format = "%.6f")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--geo_dir', type = str, default = "../geo/")
    parser.add_argument("--tag", type = str, default = "i90")
    parser.add_argument("--angle", type = float, default = 0)
    parser.add_argument("--epsg", type = int, default = 3528)
    args = parser.parse_args()

    transform_local(**vars(args))


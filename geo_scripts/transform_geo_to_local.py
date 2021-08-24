#!/Users/jsaxon/anaconda/envs/py-geo/bin/python

import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('Agg')

from fiona.crs import from_epsg
from shapely.geometry import Point

import argparse


def transform_local(geog, angle, epsg):

    df = pd.read_csv(geog)
    
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
    gdf.loc[gdf.id == "CAM"].plot(color = "red", ax = ax)
    ax.figure.savefig(geog.replace(".csv", ".pdf"),
                      bbox_inches = "tight", pad_inches = 0.1)
    
    gdf = gdf[["id", "lon", "lat", "x", "y", "xp", "yp"]].copy()

    gdf.to_csv(geog, index = False, float_format = "%.6f")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get a local projection.')
    parser.add_argument('--geog', type = str, default = "../geo/i90.csv")
    parser.add_argument("--angle", type = float, default = 0)
    parser.add_argument("--epsg", type = int, default = 3528)
    args = parser.parse_args()

    transform_local(**vars(args))


"""Apply fitted Kriging model to calculate gridded gauge-radar correction
factors.

Input
-----
- Kriging model file generated by running fit_kriging_model.py

Output
------
The output is two arrays:
  1: gridded gauge-radar correction factors corr = log10(gauge / radar)
  2: Kriging variance of the correction factors

If regression-Kriging is used, the variance is not estimated and its set to
zero.

Two output formats are supported: GeoTIFF and compressed npz file.

Configuration files (in the config/<profile> directory)
-------------------------------------------------------
- compute_kriged_correction_factors.cfg
- radar_locations.yaml

Notes
-----
When applying this script, use a model fitted to data that is close to the
target interpolation time. Also, if any data locations or values change, you
need to re-run fit_kriging_model.py.
"""

import argparse
import configparser
from datetime import datetime
import os
import pickle

import numpy as np
import pyproj
import yaml

import exporters
import util

# parse command-line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("model", type=str, help="Kriging model file")
argparser.add_argument("outtime", type=str, help="time stamp for output (YYYYmmddHHMM)")
argparser.add_argument("outfile", type=str, help="output file (without extension)")
argparser.add_argument("profile", type=str, help="configuration profile to use")
args = argparser.parse_args()

# read configuration file
config = configparser.ConfigParser()
config.read(
    os.path.join("config", args.profile, "compute_kriged_correction_factors.cfg")
)

if config["kriging"]["method"] not in ["ordinary", "regression"]:
    raise ValueError(
        f"unsupported Kriging method {config['kriging']['method']}: choose 'ordinary' or 'regression'"
    )

if config["kriging"]["method"] == "regression":
    with open(os.path.join("config", args.profile, "radar_locations.yaml"), "r") as f:
        config_radarlocs = yaml.safe_load(f)
    radar_locs = util.read_radar_locations(config_radarlocs)

# read Kriging model
model = pickle.load(open(args.model, "rb"))

projection = config["grid"]["projection"]

# read configuration and initialize the output grid
ll_lon = float(config["grid"]["ll_lon"])
ll_lat = float(config["grid"]["ll_lat"])
ur_lon = float(config["grid"]["ur_lon"])
ur_lat = float(config["grid"]["ur_lat"])

n_pixels_x = int(config["grid"]["n_pixels_x"])
n_pixels_y = int(config["grid"]["n_pixels_y"])

pr = pyproj.Proj(projection)
ll_x, ll_y = pr(ll_lon, ll_lat)
ur_x, ur_y = pr(ur_lon, ur_lat)

grid_x = np.linspace(ll_x, ur_x, n_pixels_x + 1)
grid_x += 0.5 * (grid_x[1] - grid_x[0])
grid_x = grid_x[:-1]

grid_y = np.linspace(ll_y, ur_y, n_pixels_y + 1)
grid_y += 0.5 * (grid_y[1] - grid_y[0])
grid_y = grid_y[:-1]

ts = datetime.strptime(args.outtime, "%Y%m%d%H%M")
grid_z = np.ones((1,)) * ts.timestamp()

if config["kriging"]["method"] == "ordinary":
    zvalues, sigmasq = model.execute("grid", grid_x, grid_y, grid_z)

    zvalues = zvalues[0, :]
    sigmasq = sigmasq[0, :]

    zvalues.set_fill_value(np.nan)
    sigmasq.set_fill_value(np.nan)
else:
    # project radar locations to grid coordinates
    radar_xy = {}
    for radar in radar_locs.keys():
        x, y = pr(radar_locs[radar][0], radar_locs[radar][1])
        radar_xy[radar] = (x, y)

    # compute gridded distances to the nearest radar for the regression model
    dist_grid = util.compute_gridded_distances_to_nearest_radar(
        ll_x,
        ll_y,
        ur_x,
        ur_y,
        int(config["grid"]["n_pixels_x"]),
        int(config["grid"]["n_pixels_y"]),
        radar_xy,
    )

    p = dist_grid.flatten()[:, np.newaxis]
    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
    xp = np.column_stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()])
    zvalues = model.predict(p, xp).reshape(grid_x.shape)
    sigmasq = np.zeros(zvalues.shape)

    zvalues = zvalues[:, :, 0]
    sigmasq = sigmasq[:, :, 0]

if config["output"]["type"] == "geotiff":
    pr = pyproj.Proj(config["grid"]["projection"])

    ll_x, ll_y = pr(config["grid"]["ll_lon"], config["grid"]["ll_lat"])
    ur_x, ur_y = pr(config["grid"]["ur_lon"], config["grid"]["ur_lat"])

    bounds = [ll_x, ll_y, ur_x, ur_y]
    out_rasters = np.stack([zvalues, sigmasq])
    exporters.export_geotiff(
        args.outfile, out_rasters, config["grid"]["projection"], bounds
    )
elif config["output"]["type"] == "numpy":
    np.savez_compressed(args.outfile, corr=zvalues.filled(), corr_var=sigmasq.filled())
else:
    raise ValueError(
        f"Output format {config['output']['type']} not supported. The valid options are 'geotiff' and 'numpy'"
    )

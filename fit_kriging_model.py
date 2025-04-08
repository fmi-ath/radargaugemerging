"""Fit a spatiotemporal Kriging model to the given points containing radar- and
gauge-measured rainfall accumulations. This implementation uses methods
implemented in PyKrige (https://geostat-framework.readthedocs.io/projects/pykrige/en/stable)
so that the third dimension is reserved for time. The fitting is done to
the variable log10(gauge / radar). Additional variables may be included when
using regression-Kriging.

Input
-----
- radar-gauge pair file generated by running collect_radar_gauge_pairs.py with
  attribute "gauge_location" included. When using regression_kriging, the
  pairs are also expected to contain the "distance_to_radar" attribute.

Output
------
Pickle dump containing the fitted model.

Configuration files (in the config/<profile> directory)
-------------------------------------------------------
- fit_kriging_model.cfg
"""

import argparse
import configparser
import os
import pickle

import numpy as np
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.rk import RegressionKriging

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_IMPORTED = True
except ImportError:
    SKLEARN_IMPORTED = False

# parse command-line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("rgpairfile", type=str, help="radar-gauge pair file")
argparser.add_argument("outfile", type=str, help="output file")
argparser.add_argument("profile", type=str, help="configuration profile to use")
args = argparser.parse_args()

# read configuration file
config = configparser.ConfigParser()
config.read(os.path.join("config", args.profile, "fit_kriging_model.cfg"))

if config["kriging"]["method"] not in ["ordinary", "regression"]:
    raise ValueError(
        f"unsupported Kriging method {config['kriging']['method']}: choose 'ordinary' or 'regression'"
    )

radar_gauge_pairs = pickle.load(open(args.rgpairfile, "rb"))

# collect radar-gauge pairs for fitting the model
x = []
y = []
z = []
val = []

for timestamp in radar_gauge_pairs.keys():
    for fmisid in radar_gauge_pairs[timestamp].keys():
        p = radar_gauge_pairs[timestamp][fmisid]
        x_, y_ = p[2]["gauge_location"]

        x.append(x_)
        y.append(y_)
        z.append(timestamp.timestamp())
        val.append(np.log10(p[1] / p[0]))

if config["kriging"]["time_scaling_factor"] == "auto":
    # a heuristic value to relate the standard deviations of the
    # spatial coordinates and timestamps to each other
    anisotropy_scaling_z = 0.5 * (np.std(x) + np.std(y)) / np.std(z)
else:
    anisotropy_scaling_z = float(config["kriging"]["time_scaling_factor"])

if config["kriging"]["method"] == "ordinary":
    model = OrdinaryKriging3D(
        x,
        y,
        z,
        val,
        variogram_model="exponential",
        anisotropy_scaling_z=anisotropy_scaling_z,
        verbose=True,
    )
else:
    if not SKLEARN_IMPORTED:
        raise ModuleNotFoundError(
            "sklearn needed for fitting regression models not found"
        )

    dists = []
    for timestamp in radar_gauge_pairs.keys():
        for fmisid in radar_gauge_pairs[timestamp].keys():
            dists.append(radar_gauge_pairs[timestamp][fmisid][2]["distance_to_radar"])

    regression_model = LinearRegression()
    model = RegressionKriging(
        regression_model=regression_model,
        method="ordinary3d",
        variogram_model="exponential",
        anisotropy_scaling=(1, anisotropy_scaling_z),
        n_closest_points=None,
        verbose=True,
    )
    model.fit(np.array(dists)[:, np.newaxis], np.column_stack([x, y, z]), val)

pickle.dump(model, open(args.outfile, "wb"))

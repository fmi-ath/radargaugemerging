"""Miscellaneous utility functions."""

from collections import defaultdict
from datetime import datetime, timedelta
import requests

import numpy as np
from scipy.spatial import KDTree


def compute_gauge_accumulations(
    gauge_obs,
    obs_accum_period,
    target_accum_period,
    timestep,
):
    """Compute accumulated rainfall from gauge observations read by using
    query_rain_gauges. Time periods with one or more missing observations are
    skipped.

    Parameters
    ----------
    gauge_obs : list
        List of gauge observation tuples. See the output of query_rain_gauges.
    obs_accum_period : int
        Accumulation period of the observations (minutes).
    target_accum_period : int
        Length of the accumulation period in the output (minutes).
    timestep : int
        Time step between gauge observations (minutes).

    Returns
    -------
    out : list
        List of triplets having the same elements as the output of
        query_rain_gauges.
    """
    gauge_obs_dict = defaultdict(dict)
    for g in gauge_obs:
        gauge_obs_dict[g[1]][g[0]] = g[2]

    out = []

    multiplier = timestep / obs_accum_period

    for sid in gauge_obs_dict.keys():
        startdate = min(gauge_obs_dict[sid].keys())
        enddate = max(gauge_obs_dict[sid].keys())

        curdate = startdate
        while curdate <= enddate:
            curdate_window = curdate - timedelta(minutes=target_accum_period - timestep)
            missing_data = False
            accum = 0

            while curdate_window <= curdate:
                if curdate_window in gauge_obs_dict[sid].keys():
                    v = gauge_obs_dict[sid][curdate_window]
                    if np.isfinite(v):
                        accum += v * multiplier
                    else:
                        missing_data = True
                else:
                    missing_data = True

                curdate_window = curdate_window + timedelta(minutes=timestep)

            if not missing_data:
                out.append((curdate, sid, accum))

            curdate = curdate + timedelta(minutes=timestep)

    return out


def compute_mask_boundary_weights(mask, max_dist):
    """Compute smooth weights around a boolean image mask by applying
    the logistic function.

    Parameters
    ----------
    mask : array_like
        Boolean mask with True/False corresponding to valid/invalid pixels.
    max_dist : float
        Maximum distance from the nearest valid pixel for nonzero weights.

    Returns
    -------
    out : numpy.ndarray
        Weights around nonzero values in mask with decreasing value as a
        function of distance. Values are in the range [0, 1].
    """
    weights = np.zeros(mask.shape)

    coords = np.where(mask)

    tree = KDTree(np.column_stack([coords[0], coords[1]]))
    coords_compl = np.where(~mask)
    d, _ = tree.query(
        np.column_stack([coords_compl[0], coords_compl[1]]),
        k=1,
        distance_upper_bound=max_dist,
    )

    weights[mask] = 1

    def f(x):
        out = np.zeros(x.shape)
        mask = x <= max_dist
        out[mask] = 1 / (1 + np.exp(-x[mask]))

        return out

    weights[coords_compl[0], coords_compl[1]] = f(-(d - 0.5 * max_dist) / max_dist * 10)

    return weights


def compute_distance_to_nearest_radar(gauge_loc, radar_locs):
    """Compute the distance of the given gauge location to the nearest radar.

    Parameters
    ----------
    gauge_loc : array_like
        x- and y-coordinates of the gauge
    radar_locs : dict
        Dictionary with tuples containing x- and y-coordinates of radars.

    Returns
    -------
    out : float
        Distance to the nearest radar.
    """
    dists = [
        np.linalg.norm(np.array(gauge_loc) - np.array(radar_locs[k])) / 1000
        for k in radar_locs.keys()
    ]
    return np.min(dists)


def compute_gridded_distances_to_nearest_points(
    grid_ll_x, grid_ll_y, grid_ur_x, grid_ur_y, n_pixels_x, n_pixels_y, point_locs
):
    """Compute distance of the given location to the nearest radar in a grid.

    Parameters
    ----------
    grid_ll_x : float
        X-coordinate of the lower-left corner of the grid.
    grid_ll_y : float
        Y-coordinate of the lower-left corner of the grid.
    grid_ur_x : float
        X-coordinate of the upper-right corner of the grid.
    grid_ur_y : float
        Y-coordinate of the upper-right corner of the grid.
    n_pixels_x : int
        Number of grid pixels in x-direction.
    n_pixels_y : int
        Number of grid pixels in y-direction.
    point_locs : dict
        Dictionary containing (x, y) tuples of point locations.

    Returns
    -------
    out : numpy.ndarray
        Gridded distances to the nearest points in point_locs.
    """
    x = np.linspace(grid_ll_x, grid_ur_x, n_pixels_x + 1)[:-1]
    x += 0.5 * (x[1] - x[0])
    y = np.linspace(grid_ll_y, grid_ur_y, n_pixels_y + 1)[:-1]
    y += 0.5 * (y[1] - y[0])
    grid_x, grid_y = np.meshgrid(x, y)

    dist_grid = np.ones(grid_x.shape) * np.inf
    for k in point_locs.keys():
        dx = np.array(point_locs[k][0]) - grid_x
        dy = np.array(point_locs[k][1]) - grid_y
        dist_grid_cur = np.sqrt(dx * dx + dy * dy) / 1000.0
        dist_grid = np.minimum(dist_grid, dist_grid_cur)

    return dist_grid


def query_rain_gauges(
    startdate, enddate, config, ll_lon=None, ll_lat=None, ur_lon=None, ur_lat=None
):
    """Query rain gauge observations and the corresponding gauge locations from
    SmartMet in the given date range.

    Parameters
    ----------
    startdate : datetime.datetime
        Start date for querying the gauge observations.
    enddate : datetime.datetime
        End date for querying the gauge observations.
    config : dict
        Configuration dictionary read from datasources.cfg, gauge subsection.
    ll_lon, ll_lat, ur_lon, ur_lat : float
        Bounding box coordinates. Gauges outside the box are not included.

    Returns
    -------
    out : tuple
        Two-element tuple containing gauge locations and gauge observations.
    """
    payload = {
        "bbox": "18.6,57.93,34.903,69.005",
        "producer": "observations_fmi",
        "param": "stationname,"
        "fmisid,"
        "utctime,"
        "latitude,"
        "longitude," + config["gauge_type"],
        "starttime": datetime.strftime(startdate - timedelta(hours=3), "%Y%m%dT%H%M"),
        "endtime": datetime.strftime(enddate + timedelta(hours=3), "%Y%m%dT%H%M"),
        "timestep": "data",
        "format": "json",
    }

    result = requests.get("http://smartmet.fmi.fi/timeseries", params=payload).json()

    gauge_lonlat = set()
    gauge_obs = []
    for i, r in enumerate(result):
        obstime = datetime.strptime(r["utctime"], "%Y%m%dT%H%M%S")
        if obstime < startdate or obstime > enddate:
            continue
        fmisids = r["fmisid"].strip("[").strip("]").split(" ")
        longitudes = [float(v) for v in r["longitude"].strip("[").strip("]").split(" ")]
        latitudes = [float(v) for v in r["latitude"].strip("[").strip("]").split(" ")]
        observations = [
            float(v) for v in r[config["gauge_type"]].strip("[").strip("]").split(" ")
        ]
        for fmisid, lon, lat, obs in zip(fmisids, longitudes, latitudes, observations):
            if fmisid != "nan":
                if ll_lon is not None and lon < ll_lon:
                    continue
                if ll_lat is not None and lat < ll_lat:
                    continue
                if ur_lon is not None and lon > ur_lon:
                    continue
                if ur_lat is not None and lat > ur_lat:
                    continue

                if np.isfinite(obs):
                    gauge_lonlat.add((fmisid, lon, lat))
                    gauge_obs.append((obstime, fmisid, obs))

    return gauge_lonlat, gauge_obs


def read_radar_locations(config):
    """Read radar locations from configuration file.

    Parameters
    ----------
    config : dict
        Dictionary read from radar_locations.yaml.

    Returns
    -------
    out : dict
        Dictionary containing radar longitudes and latitudes.
    """
    out = {}
    for radar in config.keys():
        out[radar] = tuple([float(v) for v in config[radar].split(",")])

    return out

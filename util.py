"""Miscellaneous utility functions."""

from datetime import datetime, timedelta
import requests

import numpy as np


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

"""For the given time range, fetch radar- and gauge-based rainfall accumulations
from the FMI archives and databases and collect co-located observation pairs.
Write the pairs into a pickle dump.

Input
-----
- Archived radar composites
- Gauge-measured rainfall accumulations from SmartMet

Output
------
Pairs of co-located radar- and gauge-based rainfall measurements. The output
file is a pickle dump containing a dictionary of the form

  radar_gauge_pairs[timestamp][fmisid]: [(r_obs_1, g_obs_1),...,(r_obs_n, g_obs_n)]

where timestamp is a datetime object defining the common time stamp for the
radar-gauge pairs. This time stamp is taken as the end time of the accumulation
period. The dictionary contains lists of co-located radar-gauge accumulation
pairs for each time in the range specified by the command-line arguments.

Configuration files (in the config/<profile> directory)
-------------------------------------------------------
- collect_radar_gauge_pairs.cfg
- datasources.cfg
- radar_locations.yaml
"""

import argparse
from collections import defaultdict
import configparser
from datetime import datetime, timedelta
import os
import pickle
import yaml
import numpy as np
import pyproj

import importers
import radar_archive
import util

# parse command-line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("startdate", type=str, help="start date (YYYYMMDDHHMM)")
argparser.add_argument("enddate", type=str, help="end date (YYYYMMDDHHMM)")
argparser.add_argument("outfile", type=str, help="output file")
argparser.add_argument("profile", type=str, help="configuration profile to use")
args = argparser.parse_args()

startdate = datetime.strptime(args.startdate, "%Y%m%d%H%M")
enddate = datetime.strptime(args.enddate, "%Y%m%d%H%M")

# read the configuration files
config = configparser.ConfigParser()
config.read(os.path.join("config", args.profile, "collect_radar_gauge_pairs.cfg"))

config_ds = configparser.ConfigParser(interpolation=None)
config_ds.read(os.path.join("config", args.profile, "datasources.cfg"))

config_radar = config_ds["radar"]
config_gauge = config_ds["gauge"]

with open(os.path.join("config", args.profile, "radar_locations.yaml"), "r") as f:
    config_radarlocs = yaml.safe_load(f)

radar_locs = util.read_radar_locations(config_radarlocs)

radar_timestep = int(config_radar["timestep"])
radar_accum_period = int(config_ds["radar"]["accum_period"])
gauge_accum_period = int(config_ds["gauge"]["accum_period"])

browser = radar_archive.Browser(
    config_radar["root_path"],
    config_radar["path_fmt"],
    config_radar["fn_pattern"],
    config_radar["fn_ext"],
    radar_timestep,
)

# read radar file names from the archive
curdate = startdate
radar_filenames = {}

while curdate <= enddate:
    try:
        fn = browser.listfiles(curdate)

        if os.path.exists(fn[0][0]):
            radar_filenames[fn[1][0]] = fn[0][0]
            # print("Found input file %s." % os.path.basename(fn[0][0]))
    except FileNotFoundError:
        pass

    curdate += timedelta(minutes=radar_accum_period)

# read gauge observations from SmartMet
cols = ["lpnn", "lat", "lat_sec", "lon", "lon_sec", "grlat", "grlon", "nvl(elstat,0)"]
col_names = ["lpnn", "lat", "lat_sec", "lon", "lon_sec", "grlat", "grlon", "elstat"]

print("Querying gauge observations from SmartMet: ", end="", flush=True)

gauge_lonlat, gauge_obs = util.query_rain_gauges(
    startdate,
    enddate,
    config_gauge,
    ll_lon=float(config["bbox"]["ll_lon"]),
    ll_lat=float(config["bbox"]["ll_lat"]),
    ur_lon=float(config["bbox"]["ur_lon"]),
    ur_lat=float(config["bbox"]["ur_lat"]),
)

# convert the lon-lat coordinates into grid coordinates (pixels)
pr = pyproj.Proj(config_radar["projection"])

x1, y1 = pr(config_radar["bbox_ll_lon"], config_radar["bbox_ll_lat"])
x2, y2 = pr(config_radar["bbox_ur_lon"], config_radar["bbox_ur_lat"])

# insert gauge locations (in grid coordinates and normalized to range [0,1])
# into a dictionary
gauge_xy = {}
gauge_xy_n = {}
for g in gauge_lonlat:
    x, y = pr(g[1], g[2])
    gauge_xy[g[0]] = (x, y)
    x = (x - x1) / (x2 - x1)
    y = (y2 - y) / (y2 - y1)
    gauge_xy_n[g[0]] = (x, y)

print(f"{len(gauge_obs)} observations from {len(gauge_xy)} gauges found.")

# insert gauge observations into a dictionary
gauge_obs_ = defaultdict(list)
for r in gauge_obs:
    gauge_obs_[r[0]].append(r[1:])
gauge_obs = gauge_obs_

r_thr = float(config["thresholds"]["radar"])
g_thr = float(config["thresholds"]["gauge"])

radar_gauge_pairs = defaultdict(dict)

rgpair_attribs = config["other"]["attributes"].split(",")

gauge_lonlats = dict([(v[0], (v[1], v[2])) for v in gauge_lonlat])


def _compute_nearest_distance(gauge_lonlat):
    pass


# collect radar-gauge observation pairs
radar_ts = startdate
while radar_ts <= enddate:
    print(
        f"Collecting radar-gauge pairs between {enddate - timedelta(minutes=gauge_accum_period)} - {enddate}:"
    )

    importer = importers.get_method(config_radar["importer"])

    # Read radar measurements from the gauge accumulation period.
    # Here we assume that the gauge values represent accumulation from the
    # previous n minutes.
    num_accum_timesteps = gauge_accum_period / radar_accum_period
    if int(num_accum_timesteps) != num_accum_timesteps:
        raise ValueError(
            "gauge accumulation period not divisible by radar accumulation period"
        )
    num_accum_timesteps = int(num_accum_timesteps)
    num_missing = 0
    num_found = 0
    radar_rain_accum_cur = 0.0

    for t in range(num_accum_timesteps):
        prev_radar_ts = radar_ts - t * timedelta(minutes=radar_accum_period)
        if not prev_radar_ts in radar_filenames.keys():
            num_missing += 1
        else:
            radar_rain_rate, _ = importer(radar_filenames[prev_radar_ts])
            radar_rain_accum_cur += radar_rain_rate
            num_found += 1

    if num_missing > int(config["missing_values"]["max_missing_radar_timestamps"]):
        print(
            f"  Skipping {radar_ts}: not enough previous files found for computing accumulated radar rainfall."
        )
    else:
        print(
            f"  Computed radar accumulation between {prev_radar_ts} - {radar_ts} from {num_found} time stamps."
        )
        radar_rain_accum_cur /= num_found
        radar_rain_accum_shape = radar_rain_accum_cur.shape

        if radar_ts in gauge_obs.keys():
            num_radar_gauge_pairs = 0
            g_cur = gauge_obs[radar_ts]
            for g in g_cur:
                fmisid = g[0]
                x, y = gauge_xy_n[fmisid][0], gauge_xy_n[fmisid][1]
                x_ = int(np.floor(x * radar_rain_accum_shape[1]))
                y_ = int(np.floor(y * radar_rain_accum_shape[0]))
                if (
                    x_ >= 0.0
                    and y_ >= 0.0
                    and x_ < radar_rain_accum_shape[1]
                    and y_ < radar_rain_accum_shape[0]
                ):
                    r_obs = radar_rain_accum_cur[y_, x_]
                    g_obs = g[1]

                    attrs = {}

                    # TODO: implement this
                    if "distance" in rgpair_attribs:
                        attrs["distance"] = _compute_nearest_distance(
                            gauge_lonlats[fmisid]
                        )

                    if "gauge_location" in rgpair_attribs:
                        attrs["gauge_location"] = (x, y)

                    if r_obs >= r_thr and g_obs >= g_thr:
                        radar_gauge_pairs[radar_ts][int(fmisid)] = (
                            r_obs,
                            g_obs,
                            *attrs,
                        )
                        num_radar_gauge_pairs += 1

            print(f"  Collected {num_radar_gauge_pairs} radar-gauge pairs.")

    radar_ts += timedelta(minutes=gauge_accum_period)

mae = 0.0
n = 0
for p1 in radar_gauge_pairs.values():
    for p2 in p1.values():
        mae += abs(p2[0] - p2[1])
        n += 1

mae = mae / n if n > 0 else np.nan

print(f"Total number of radar-gauge pairs: {n}")
print(f"Mean absolute radar-gauge error: {mae}")

print(f"Wrote output to {args.outfile}.")

pickle.dump(radar_gauge_pairs, open(args.outfile, "wb"))

import gzip
from matplotlib.pyplot import imread
import numpy as np

try:
    import h5py

    H5PY_IMPORTED = True
except ImportError:
    H5PY_IMPORTED = False

try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

try:
    import xarray as xr

    XARRAY_IMPORTED = True
except ImportError:
    XARRAY_IMPORTED = False


def get_method(name):
    """Return the given importer method. The currently implemented options are
    'netcdf', 'odim_hdf5' and 'pgm'."""
    if name == "netcdf":
        return import_netcdf
    elif name == "odim_hdf5":
        return import_opera_odim_hdf5
    elif name == "pgm":
        return import_pgm
    else:
        raise NotImplementedError(f"importer {name} not implemented")


def import_netcdf(filename, corr_refl=True, **kwargs):
    """Import a NetCDF file produced by radar_composite_generator."""
    if not XARRAY_IMPORTED:
        raise ModuleNotFoundError("xarray is required but not installed")
    if not NETCDF4_IMPORTED:
        raise ModuleNotFoundError("netCDF4 is required but not installed")

    ds = xr.load_dataset(filename)
    qty = "DBZHC" if corr_refl else "DBZH"

    refl = np.array(ds[qty][0])
    precip = 10.0 ** (refl / 10.0)
    precip = (precip / 223.0) ** (1.0 / 1.53)

    return precip, dict()


def import_pgm(filename, gzipped=True, **kwargs):
    """
    Import a 8-bit PGM radar reflectivity composite from the FMI archive and
    convert it to precipitation rate (mm/h).

    Parameters
    ----------
    filename: str
        Name of the file to read.
    gzipped: bool
        If True, the input file is treated as a compressed gzip file.

    Raises
    ------
    ModuleNotFoundError
        If pyproj was not found.

    Returns
    -------
    out: tuple
        A two-element tuple containing the reflectivity composite in dBZ and
        the associated metadata.

    Notes
    -----
    Reading georeferencing metadata is supported only for stereographic
    projection. For other projections, the keys related to georeferencing are
    not set.
    """
    if not PYPROJ_IMPORTED:
        raise ModuleNotFoundError("pyproj is required but not installed")

    if not int(gzipped):
        precip = imread(filename)
    else:
        precip = imread(gzip.open(filename, "r"))
    pgm_metadata = _import_fmi_pgm_metadata(filename, gzipped=gzipped)
    geodata = _import_fmi_pgm_geodata(pgm_metadata)

    mask = precip == pgm_metadata["missingval"]
    precip = precip.astype(float)
    precip[mask] = np.nan
    precip = (precip - 64.0) / 2.0

    # convert from dBZ to rain rate (mm/h)
    precip = 10.0 ** (precip / 10.0)
    precip = (precip / 223.0) ** (1.0 / 1.53)

    metadata = geodata
    metadata["institution"] = "Finnish Meteorological Institute"
    metadata["accutime"] = 5.0
    metadata["unit"] = "dBZ"
    metadata["transform"] = "dB"
    metadata["zerovalue"] = np.nanmin(precip)
    metadata["threshold"] = _get_threshold_value(precip)
    metadata["zr_a"] = 223.0
    metadata["zr_b"] = 1.53

    return precip, metadata


def import_opera_odim_hdf5(filename, quantity="DBZH", **kwargs):
    """Read a composite from a OPERA ODIM HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the file to read.
    quantity : the quantity to read
        ACRR: hourly accumulated rainfall (mm)
        DBZH: reflectivity (dBZ)
        RATE: rainfall rate (mm/h)

    Raises
    ------
    KeyError
        If the requested quantity was not found.
    ModuleNotFoundError
        If h5py or pyproj was not found.
    OSError
        If the input file could not be read.

    Returns
    -------
    out : tuple
        A two-element tuple containing the radar composite in a numpy array and
        the metadata dictionary.
    """
    if not H5PY_IMPORTED:
        raise ModuleNotFoundError("h5py required for reading HDF5 files but not found")

    if not PYPROJ_IMPORTED:
        raise ModuleNotFoundError("pyproj is required but not installed")

    f = h5py.File(filename, "r")

    data_found = False

    for k in f.keys():
        if "dataset" in k:
            try:
                what_group = f[k]["what"]
                qty = what_group.attrs["quantity"]
                nodata = what_group.attrs["nodata"]
                undetect = what_group.attrs["undetect"]
                gain = what_group.attrs["gain"]
                offset = what_group.attrs["offset"]
            except KeyError:
                what_group = f[k]["data1"]["what"]
                qty = what_group.attrs["quantity"]
                nodata = what_group.attrs["nodata"]
                undetect = what_group.attrs["undetect"]
                gain = what_group.attrs["gain"]
                offset = what_group.attrs["offset"]

            if qty.decode() == quantity:
                data_found = True

                data = f[k]["data1"]["data"][...]
                nodata_mask = data == nodata
                undetect_mask = data == undetect

                radar_composite = data.astype(np.float32)
                radar_composite = radar_composite * gain + offset

                radar_composite[nodata_mask] = np.nan

                if quantity != "DBZH":
                    radar_composite[undetect_mask] = offset
                else:
                    radar_composite[undetect_mask] = -30.0

                metadata = {}
                projection = f["where"].attrs["projdef"].decode()

                metadata["projection"] = projection

                ll_lon = f["where"].attrs["LL_lon"]
                ll_lat = f["where"].attrs["LL_lat"]
                ur_lon = f["where"].attrs["UR_lon"]
                ur_lat = f["where"].attrs["UR_lat"]

                pr = pyproj.Proj(projection)
                ll_x, ll_y = pr(ll_lon, ll_lat)
                ur_x, ur_y = pr(ur_lon, ur_lat)

                metadata["ll_x"] = ll_x
                metadata["ll_y"] = ll_y
                metadata["ur_x"] = ur_x
                metadata["ur_y"] = ur_y

                xpixelsize = f["where"].attrs["xscale"]
                ypixelsize = f["where"].attrs["yscale"]

                metadata["xpixelsize"] = xpixelsize
                metadata["ypixelsize"] = ypixelsize

                metadata["institution"] = "Finnish Meteorological Institute"
                metadata["timestep"] = 5
                if quantity == "ACRR":
                    metadata["unit"] = "mm"
                elif quantity == "DBZH":
                    metadata["unit"] = "dBZ"
                elif quantity == "RATE":
                    metadata["unit"] = "mm/h"

                break

    f.close()

    if not data_found:
        raise KeyError(f"no composite for quantity '{quantity}' found from {filename}")
    else:
        return radar_composite, metadata


def _import_fmi_pgm_geodata(metadata):
    geodata = {}

    projdef = ""

    if "type" in metadata.keys() and metadata["type"][0] == "stereographic":
        projdef += "+proj=stere "
        projdef += " +lon_0=" + metadata["centrallongitude"][0] + "E"
        projdef += " +lat_0=" + metadata["centrallatitude"][0] + "N"
        projdef += " +lat_ts=" + metadata["truelatitude"][0]
        # These are hard-coded because the projection definition
        # is missing from the PGM files.
        projdef += " +a=6371288"
        projdef += " +x_0=380886.310"
        projdef += " +y_0=3395677.920"
        projdef += " +no_defs"
        #
        geodata["projection"] = projdef

        ll_lon, ll_lat = [float(v) for v in metadata["bottomleft"]]
        ur_lon, ur_lat = [float(v) for v in metadata["topright"]]

        pr = pyproj.Proj(projdef)
        x1, y1 = pr(ll_lon, ll_lat)
        x2, y2 = pr(ur_lon, ur_lat)

        geodata["x1"] = x1
        geodata["y1"] = y1
        geodata["x2"] = x2
        geodata["y2"] = y2
        geodata["cartesian_unit"] = "m"
        geodata["xpixelsize"] = float(metadata["metersperpixel_x"][0])
        geodata["ypixelsize"] = float(metadata["metersperpixel_y"][0])

        geodata["yorigin"] = "upper"

    return geodata


def _get_threshold_value(precip):
    valid_mask = np.isfinite(precip)
    if valid_mask.any():
        _precip = precip[valid_mask]
        min_precip = _precip.min()
        above_min_mask = _precip > min_precip
        if above_min_mask.any():
            return np.min(_precip[above_min_mask])
        else:
            return min_precip
    else:
        return np.nan


def _import_fmi_pgm_metadata(filename, gzipped=False):
    metadata = {}

    if not int(gzipped):
        f = open(filename, "rb")
    else:
        f = gzip.open(filename, "rb")

    file_line = f.readline()
    while not file_line.startswith(b"#"):
        file_line = f.readline()
    while file_line.startswith(b"#"):
        x = file_line.decode()
        x = x[1:].strip().split(" ")
        if len(x) >= 2:
            k = x[0]
            v = x[1:]
            metadata[k] = v
        else:
            file_line = f.readline()
            continue
        file_line = f.readline()
    file_line = f.readline().decode()
    metadata["missingval"] = int(file_line)
    f.close()

    return metadata

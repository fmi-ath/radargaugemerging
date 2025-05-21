"""Methods for writing output files."""

from osgeo import gdal, osr


def export_geotiff(filename, rasters, projection, bounds):
    """Write rasters to a GeoTIFF file.

    Parameters
    ----------
    filename : str
      Output file name.
    rasters : array_like
      Three-dimensional array of shape (channels,height,width) containing the
      rasters to write.
    projection : str
      PROJ-compatible projection definition.
    bounds : list or tuple
      List or tuple (x_min,y_min,x_max,y_max) defining the geographical bounds
      of the rasters.
    """
    driver = gdal.GetDriverByName("GTiff")

    dst = driver.Create(
        filename,
        rasters.shape[2],
        rasters.shape[1],
        rasters.shape[0],
        gdal.GDT_Float32,
        ["COMPRESS=DEFLATE", "PREDICTOR=3"],
    )

    srs = osr.SpatialReference()
    srs.ImportFromProj4(projection)
    dst.SetProjection(srs.ExportToWkt())

    xmin, ymin, xmax, ymax = bounds

    ny = rasters.shape[1]
    nx = rasters.shape[2]
    xres = (xmax - xmin) / nx
    yres = (ymax - ymin) / ny
    geotransform = (xmin, xres, 0, ymin, 0, yres)
    dst.SetGeoTransform(geotransform)

    for i in range(rasters.shape[0]):
        dst.GetRasterBand(i + 1).WriteArray(rasters[i])

    dst.FlushCache()

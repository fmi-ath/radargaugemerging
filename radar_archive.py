"""Utilities for browsing archived radar composites."""

from datetime import datetime, timedelta
import fnmatch
import os


class Browser:
    """Implements a browser for listing archives radar composites that match the
    given date."""

    def __init__(self, root_path, path_fmt, fn_pattern, fn_exts, timestep):
        """Create a radar archive browser.

        Parameters
        ----------
        root_path : str
          The root path for the input files.
        path_fmt : str
          Path format. May contain directory names separated by '/' and date/time
          specifiers beginning with '%' (e.g. %Y/%m/%d).
        fn_pattern : str
          The name pattern of the input files without extension. May contain time
          specifiers (e.g. %H, %M and %S).
        fn_exts : list
          List of possible input file name extensions.
        timestep : float
          Time step between consecutive input files (minutes).
        """
        self.__root_path = root_path
        self.__path_fmt = path_fmt
        self.__fn_pattern = fn_pattern
        self.__fn_exts = fn_exts
        self.__timestep = timestep

    def listfiles(self, date, num_prev_files=0):
        """List input files whose timestamp matches the given date.

        Parameters
        ----------
        date : datetime.datetime
          The given date.
        num_prev_files : int
          Optional, number of previous files to find before the given timestamp.

        Returns
        -------
        out : tuple
          If num_prev_files=0, return a pair containing the found file name and the
          corresponding timestamp as a datetime.datetime object. Otherwise, return a
          tuple of two lists, the first one for the file names and the second one
          for the corresponding timestemps. The lists are sorted in ascending order
          with respect to timestamp.
        """
        return _find_by_date(
            date,
            self.__root_path,
            self.__path_fmt,
            self.__fn_pattern,
            self.__fn_exts,
            self.__timestep,
            num_prev_files=num_prev_files,
        )


def _find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_exts, timestep, num_prev_files=0
):
    filenames = []
    timestamps = []

    for i in range(num_prev_files + 1):
        curdate = date - timedelta(minutes=i * timestep)
        fn = _find_matching_filename(curdate, root_path, path_fmt, fn_pattern, fn_exts)
        filenames.append(fn)

        timestamps.append(curdate)

    if all(filename is None for filename in filenames):
        raise FileNotFoundError("no input data found in %s" % root_path)

    if (num_prev_files) > 0:
        return (filenames[::-1], timestamps[::-1])
    else:
        return (filenames, timestamps)


def _find_matching_filename(date, root_path, path_fmt, fn_pattern, fn_exts):
    path = _generate_path(date, root_path, path_fmt)
    fn = None

    if os.path.exists(path):
        for fn_ext in fn_exts:
            fn = datetime.strftime(date, fn_pattern) + "." + fn_ext

            # test for wildcards
            if "?" in fn:
                filenames = os.listdir(path)
                if len(filenames) > 0:
                    for filename in filenames:
                        if fnmatch.fnmatch(filename, fn):
                            fn = filename
                            break

            fn = os.path.join(path, fn)
            if os.path.exists(fn):
                return fn
            else:
                fn = None

    return fn


def _generate_path(date, root_path, path_fmt):
    f = lambda t: datetime.strftime(date, t) if t[0] == "%" else t
    if path_fmt != "":
        tokens = [f(t) for t in path_fmt.split("/")]
        subpath = os.path.join(*tokens)

        return os.path.join(root_path, subpath)
    else:
        return root_path

"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Functionalities for the import and basic processing of ISIMIP output data (.nc)
Data to be downloaded from https://esg.pik-potsdam.de/search/isimip/
(requires ESG login, sript will be added later)


All functions should work for ISIMIP netcdf output files regardless of the resolution
of saptial (lat, lon) and temporal (time) resolution and dimension of input data.
Not that ISIMIP data comes in a range of resolutions, i.e. daily (e.g. discharge),
monthly, yearly (e.g. yield)
"""


import xarray as xr

bbox_world = [-85, 85, -180, 180]

def _read_one_nc(file_name, bbox=None, years=None):
    """Reads 1 ISIMIP output NETCDF file data within a certain bounding box and time period

    Parameters
    ----------
    file_name : str
        Absolute or relative path to *.nc
    bbox : array
        bounding box containing [Lon min, lat min, lon max, lat max]
    years : array
        start and end year of the time series that shall be extracted

    Returns
    -------
    data : dataset
        Contains data in the specified bounding box and for the
        specified time period
    """
    data = xr.open_dataset(file_name, decode_times=False)
    if not bbox:
        bbox = bbox_world
    if not years:
        return data.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]))

    time_id = years - int(data.time.units[12:16])
    return data.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]),
                    time=slice(time_id[0], time_id[1]))

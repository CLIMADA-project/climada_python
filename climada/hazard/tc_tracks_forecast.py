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

Define TCTracks auxiliary methods: BUFR based TC predictions (from ECMWF)
"""

__all__ = ['TCForecast']

# standard libraries
import datetime as dt
import fnmatch
import ftplib
import logging
import tempfile
from pathlib import Path
import collections

# additional libraries
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import eccodes as ec

# climada dependencies
from climada.hazard.tc_tracks import (
    TCTracks, set_category, DEF_ENV_PRESSURE, CAT_NAMES
)
from climada.util.files_handler import get_file_names

# declare constants
ECMWF_FTP = 'dissemination.ecmwf.int'
ECMWF_USER = 'wmo'
ECMWF_PASS = 'essential'

BASINS = {
    'W': 'W - North West Pacific',
    'C': 'C - North Central Pacific',
    'E': 'E - North East Pacific',
    'P': 'P - South Pacific',
    'L': 'L - North Atlantic',
    'A': 'A - Arabian Sea (North Indian Ocean)',
    'B': 'B - Bay of Bengal (North Indian Ocean)',
    'U': 'U - Australia',
    'S': 'S - South-West Indian Ocean',
    'X': 'X - Undefined Basin'
}
"""Gleaned from the ECMWF wiki at
https://confluence.ecmwf.int/display/FCST/Tropical+Cyclone+tracks+in+BUFR+-+including+genesis
with added basin 'X' to deal with it appearing in operational forecasts
(see e.g. years 2020 and 2021 in the sidebar at https://www.ecmwf.int/en/forecasts/charts/tcyclone/)
and Wikipedia at https://en.wikipedia.org/wiki/Invest_(meteorology)
"""

SAFFIR_MS_CAT = np.array([18, 33, 43, 50, 59, 71, 1000])
"""Saffir-Simpson Hurricane Categories in m/s"""

SIG_CENTRE = 1
"""The BUFR code 008005 significance for 'centre'"""

LOGGER = logging.getLogger(__name__)

MISSING_DOUBLE = ec.CODES_MISSING_DOUBLE
MISSING_LONG = ec.CODES_MISSING_LONG
"""Missing double and integers in ecCodes """

class TCForecast(TCTracks):
    """An extension of the TCTracks construct adapted to forecast tracks
    obtained from numerical weather prediction runs.

    Attributes:
        data (list(xarray.Dataset)): Same as in parent class, adding the
            following attributes
                - ensemble_member (int)
                - is_ensemble (bool; if False, the simulation is a high resolution
                               deterministic run)
    """

    def fetch_ecmwf(self, path=None, files=None):
        """
        Fetch and read latest ECMWF TC track predictions from the FTP
        dissemination server into instance. Use path argument to use local
        files instead.

        Parameters:
            path (str, list(str)): A location in the filesystem. Either a
                path to a single BUFR TC track file, or a folder containing
                only such files, or a globbing pattern. Passed to
                climada.util.files_handler.get_file_names
            files (file-like): An explicit list of file objects, bypassing
                get_file_names
        """
        if path is None and files is None:
            files = self.fetch_bufr_ftp()
        elif files is None:
            files = get_file_names(path)

        for i, file in tqdm.tqdm(enumerate(files, 1), desc='Processing',
                                 unit='files', total=len(files)):
            try:
                file.seek(0)  # reset cursor if opened file instance
            except AttributeError:
                pass

            self.read_one_bufr_tc(file, id_no=i)

            try:
                file.close()  # discard if tempfile
            except AttributeError:
                pass

    @staticmethod
    def fetch_bufr_ftp(target_dir=None, remote_dir=None):
        """
        Fetch and read latest ECMWF TC track predictions from the FTP
        dissemination server. If target_dir is set, the files get downloaded
        persistently to the given location. A list of opened file-like objects
        gets returned.

        Parameters:
            target_dir (str): An existing directory to write the files to. If
                None, the files get returned as tempfiles.
            remote_dir (str, optional): If set, search this ftp folder for
                forecast files; defaults to the latest. Format:
                yyyymmddhhmmss, e.g. 20200730120000

        Returns:
            [str] or [filelike]
        """
        con = ftplib.FTP(host=ECMWF_FTP, user=ECMWF_USER, passwd=ECMWF_PASS)

        try:
            if remote_dir is None:
                remote = pd.Series(con.nlst())
                remote = remote[remote.str.contains('120000|000000$')]
                remote = remote.sort_values(ascending=False)
                remote_dir = remote.iloc[0]

            con.cwd(remote_dir)

            remotefiles = fnmatch.filter(con.nlst(), '*tropical_cyclone*')
            if len(remotefiles) == 0:
                msg = 'No tracks found at ftp://{}/{}'
                msg.format(ECMWF_FTP, remote_dir)
                raise FileNotFoundError(msg)

            localfiles = []

            LOGGER.info('Fetching BUFR tracks:')
            for rfile in tqdm.tqdm(remotefiles, desc='Download', unit=' files'):
                if target_dir:
                    lfile = Path(target_dir, rfile).open('w+b')
                else:
                    lfile = tempfile.TemporaryFile(mode='w+b')

                con.retrbinary('RETR ' + rfile, lfile.write)

                if target_dir:
                    localfiles.append(lfile.name)
                    lfile.close()
                else:
                    localfiles.append(lfile)

        except ftplib.all_errors as err:
            con.quit()
            raise type(err)('Error while downloading BUFR TC tracks: ' + str(err)) from err

        _ = con.quit()

        return localfiles

    def read_one_bufr_tc(self, file, id_no=None):
        """ Read a single BUFR TC track file.

        Parameters:
            file (str, filelike): Path object, string, or file-like object
            id_no (int): Numerical ID; optional. Else use date + random int.
        """     
        # Open the bufr file
        bufr = ec.codes_bufr_new_from_file(file)
        # we need to instruct ecCodes to expand all the descriptors
        # i.e. unpack the data values
        ec.codes_set(bufr, 'unpack', 1)
        
        # get the forcast time
        timestamp_origin = dt.datetime(
            ec.codes_get(bufr, 'year'), ec.codes_get(bufr, 'month'),
            ec.codes_get(bufr, 'day'), ec.codes_get(bufr, 'hour'),
            ec.codes_get(bufr, 'minute'),
        )
        timestamp_origin = np.datetime64(timestamp_origin)
        
        # get storm identifier
        sid = ec.codes_get(bufr, 'stormIdentifier').strip()
        
        # number of timestep (size of the forecast time + initial timestep)
        try:
            n_timestep = ec.codes_get_size(bufr, 'timePeriod') + 1
        except ec.CodesInternalError:
            LOGGER.warning("Track %s has no defined timePeriod."
                            "Track is discarded.", sid)
            return None
            
        # ensemble members number
        ens_no = ec.codes_get_array(bufr, "ensembleMemberNumber")
        
        # values at timestep 0
        lat_init = ec.codes_get_array(bufr, '#2#latitude')
        lon_init = ec.codes_get_array(bufr, '#2#longitude')
        pre_init = ec.codes_get_array(bufr, '#1#pressureReducedToMeanSeaLevel')
        max_wind_init = ec.codes_get_array(bufr, '#1#windSpeedAt10M')
        
        if len(lat_init) == len(ens_no) and len(max_wind_init) == len(ens_no):
            latitude = {ind_ens: np.array(lat_init[ind_ens]) for ind_ens in range(len(ens_no))}
            longitude = {ind_ens: np.array(lon_init[ind_ens]) for ind_ens in range(len(ens_no))}
            pressure = {ind_ens: np.array(pre_init[ind_ens]) for ind_ens in range(len(ens_no))}
            max_wind = {ind_ens: np.array(max_wind_init[ind_ens]) for ind_ens in range(len(ens_no))}
        else:
            latitude = {ind_ens: np.array(lat_init[0]) for ind_ens in range(len(ens_no))}
            longitude = {ind_ens: np.array(lon_init[0]) for ind_ens in range(len(ens_no))}
            pressure = {ind_ens: np.array(pre_init[0]) for ind_ens in range(len(ens_no))}
            max_wind = {ind_ens: np.array(max_wind_init[0]) for ind_ens in range(len(ens_no))}
        
        # getting the forecasted storms
        timesteps_int = [0 for x in range(n_timestep)]
        for ind_timestep in range(1, n_timestep):
            rank1 = ind_timestep * 2 + 2 # rank for getting storm centre information
            rank3 = ind_timestep * 2 + 3 # rank for getting max wind information
            
            timestep = ec.codes_get_array(bufr, "#%d#timePeriod" % ind_timestep)
            if len(timestep) == 1:
                timesteps_int[ind_timestep] = timestep[0]
            else:
                for i in range(len(timestep)):
                    if timestep[i] != MISSING_LONG:
                        timesteps_int[ind_timestep] = timestep[i]
                        break
            
            # Location of the storm
            sig_values = ec.codes_get_array(bufr, "#%d#meteorologicalAttributeSignificance" % rank1)
            if len(sig_values) == 1:
                significance = sig_values[0]
            else:
                for j in range(len(sig_values)):
                    if sig_values[j] != ec.CODES_MISSING_LONG:
                        significance = sig_values[j]
                        break
            # get lat, lon, and pre of all ensemble members at ind_timestep        
            if significance == 1:
                lat_temp = ec.codes_get_array(bufr, "#%d#latitude" % rank1)
                lon_temp = ec.codes_get_array(bufr, "#%d#longitude" % rank1)
                pre_temp = ec.codes_get_array(bufr, "#%d#pressureReducedToMeanSeaLevel" % (ind_timestep + 1))
            else:
                raise ValueError('unexpected meteorologicalAttributeSignificance=', significance)
                
            # Location of max wind
            sig_values = ec.codes_get_array(bufr, "#%d#meteorologicalAttributeSignificance" % rank3)
            if len(sig_values) == 1:
                significanceWind = sig_values[0]
            else:
                for j in range(len(sig_values)):
                    if sig_values[j] != ec.CODES_MISSING_LONG:
                        significanceWind = sig_values[j]
                        break
            # max_wind of all ensemble members at ind_timestep    
            if significanceWind == 3:
                wnd_temp = ec.codes_get_array(bufr, "#%d#windSpeedAt10M" % (ind_timestep + 1))
            else:
                raise ValueError('unexpected meteorologicalAttributeSignificance=', significance)
                
            # check dimention of the variables, and replace missing value with NaN
            lat = self._check_variable(lat_temp, ens_no)
            lon = self._check_variable(lon_temp, ens_no)
            pre = self._check_variable(pre_temp, ens_no)
            wnd = self._check_variable(wnd_temp, ens_no)
            
            # appending values
            for ind_ens in range(len(ens_no)):
                latitude[ind_ens] = np.append(latitude[ind_ens], lat[ind_ens])
                longitude[ind_ens] = np.append(longitude[ind_ens], lon[ind_ens])
                pressure[ind_ens] = np.append(pressure[ind_ens], pre[ind_ens])
                max_wind[ind_ens] = np.append(max_wind[ind_ens], wnd[ind_ens])

        
        # storing information into a dictionary
        msg = {
            # subset forecast data
            'latitude': latitude,
            'longitude': longitude,
            'wind_10m': max_wind,
            'pressure': pressure,
            'timestamp': timesteps_int,

            # subset metadata
            'wmo_longname': ec.codes_get(bufr, 'longStormName').strip(),
            'storm_id': ec.codes_get(bufr, 'stormIdentifier').strip(),
            'ens_type': ec.codes_get_array(bufr, 'ensembleForecastType'),
            'ens_number': ec.codes_get_array(bufr, "ensembleMemberNumber"),
        }

        if id_no is None:
            id_no = timestamp_origin.item().strftime('%Y%m%d%H') + \
                    str(np.random.randint(1e3, 1e4))

        orig_centre = ec.codes_get(bufr, 'centre')
        if orig_centre == 98:
            provider = 'ECMWF'
        else:
            provider = 'BUFR code ' + str(orig_centre)

        for i in range(len(msg['ens_number'])):
            name = msg['wmo_longname']
            track = self._subset_to_track(
                msg, i, provider, timestamp_origin, name, id_no
            )
            if track is not None:
                self.append(track)
            else:
                LOGGER.debug('Dropping empty track %s, subset %d', name, i)

    @staticmethod
    def _subset_to_track(msg, index, provider, timestamp_origin, name, id_no):
        """Subroutine to process one BUFR subset into one xr.Dataset"""
        lat = np.array(msg['latitude'][index], dtype='float')
        lon = np.array(msg['longitude'][index], dtype='float')
        wnd = np.array(msg['wind_10m'][index], dtype='float')
        pre = np.array(msg['pressure'][index], dtype='float')

        sid = msg['storm_id'].strip()

        timestep_int = np.array(msg['timestamp']).squeeze()
        timestamp = timestamp_origin + timestep_int.astype('timedelta64[h]')
        
        # some weak storms have only perturbed analysis, which gives a 
        # size 1 array with value 4
        try:
            ens_bool = msg['ens_type'][index] != 0
        except LookupError:
            ens_bool = msg['ens_type'][0] != 0
            LOGGER.info('All tracks has the same ensemble type')
            return None
        
        try:
            track = xr.Dataset(
                data_vars={
                    'max_sustained_wind': ('time', np.squeeze(wnd)),
                    'central_pressure': ('time', np.squeeze(pre)/100),
                    'ts_int': ('time', timestep_int),
                    'lat': ('time', lat),
                    'lon': ('time', lon),
                },
                coords={
                    'time': timestamp,
                },
                attrs={
                    'max_sustained_wind_unit': 'm/s',
                    'central_pressure_unit': 'mb',
                    'name': name,
                    'sid': sid,
                    'orig_event_flag': False,
                    'data_provider': provider,
                    'id_no': (int(id_no) + index / 100),
                    'ensemble_number': msg['ens_number'][index],
                    'is_ensemble': ens_bool,
                    'forecast_time': timestamp_origin,
                }
            )
        except ValueError as err:
            LOGGER.warning(
                'Could not process track %s subset %d, error: %s',
                sid, index, err
                )
            return None

        track = track.dropna('time')

        if track.sizes['time'] == 0:
            return None

        # can only make latlon coords after dropna
        track = track.set_coords(['lat', 'lon'])
        track['time_step'] = track.ts_int - \
            track.ts_int.shift({'time': 1}, fill_value=0)

        track = track.drop_vars(['ts_int'])

        track['radius_max_wind'] = (('time'), np.full_like(
            track.time, np.nan, dtype=float)
        )
        track['environmental_pressure'] = (('time'), np.full_like(
            track.time, DEF_ENV_PRESSURE, dtype=float)
        )

        # according to specs always num-num-letter
        track['basin'] = ('time', np.full_like(track.time, BASINS[sid[2]], dtype=object))

        if sid[2] == 'X':
            LOGGER.info(
                'Undefined basin %s for track name %s ensemble no. %d',
                sid[2], track.attrs['name'], track.attrs['ensemble_number'])

        cat_name = CAT_NAMES[set_category(
            max_sus_wind=track.max_sustained_wind.values,
            wind_unit=track.max_sustained_wind_unit,
            saffir_scale=SAFFIR_MS_CAT
        )]
        track.attrs['category'] = cat_name
        return track

    @staticmethod
    def _check_variable(var, ens_no):
        """Check the value and dimension of variable"""
        if len(var) == len(ens_no):
            var[var==MISSING_DOUBLE] = np.nan
            return var
        elif len(var) == 1 and var[0] == MISSING_DOUBLE:
            return np.repeat(np.nan, len(ens_no))
        elif len(var) == 1 and var[0] != MISSING_DOUBLE:
            return np.repeat(var[0], len(ens_no))
            LOGGER.warning('Only 1 variable value for %d ensble members, duplicate value to all members',
                           len(ens_no))
        else:
            raise ValueError
            

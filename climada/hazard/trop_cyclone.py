"""
Define TropCyclone class and IBTracs reader.
"""

__all__ = ['TropCyclone', 'read_ibtracs']

import logging
import datetime as dt
import pandas as pd
import xarray as xr
import numpy as np
from pint import UnitRegistry

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.base import Centroids
from climada.util.constants import GLB_CENTROIDS
import climada.util.checker as check

LOGGER = logging.getLogger(__name__)

SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 135, 1000]

class TropCyclone(Hazard):
    """Contains events of Tropical Cyclones.

    Attributes:
        tag (TagHazard): information about the source
        units (str): units of the intensity
        centroids (Centroids): centroids of the events
        event_id (np.array): id (>0) of each event
        event_name (list): name of each event (set as event_id if no provided)
        frequency (np.array): frequency of each event in seconds
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
        track (list(xarray.Dataset)): list of tropical cyclone tracks
    """

    def __init__(self, file_name='', description='', centroids=None):
        """Initialize values from given file, if given.

        Parameters:
            file_name (str or list(str), optional): file name(s) or folder name 
                containing the files to read
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC')
            description (str or list(str), optional): one description of the
                data or a description of each data file
            centroids (Centroids or list(Centroids), optional): Centroids

        Raises:
            ValueError
        """
        self.haz_type = 'TC'
        self.track = [] # [xr.Dataset()]
        Hazard.__init__(self, file_name, self.haz_type, description, centroids)
    
    @staticmethod
    def _read_one(file_name, haz_type='TC', description='', centroids=None, \
                  var_names=None):
        """ Read input file. If centroids are not provided, they are read
        from file_name.

        Parameters:
            file_name (str): name of the source file
            haz_type (str): 'TC'
            description (str, optional): description of the source data
            centroids (Centroids, optional): Centroids instance
            var_names (dict): None

        Raises:
            ValueError, KeyError
        """
        if centroids is None:
            centroids = Centroids(GLB_CENTROIDS, 'Global Nat centroids')
        new_haz = TropCyclone()
        new_haz.tag = TagHazard(file_name, haz_type, description)
        try:
            track = read_ibtracs(file_name)
        except pd.errors.ParserError:
            LOGGER.error('Provide a IBTraCS file in csv format containing one'\
                         + ' TC track.')
            raise ValueError
        new_haz._event_from_track(track, centroids)
        new_haz.track.append(track)
        return new_haz
    
    def _event_from_track(self, track, centroids):
        """ """
        # TODO: coastal_centroids + climada_tc_equal_timestep + isimip_windfield_holland
        raise NotImplementedError
    
    def append(self, tc_haz):
        """Check and append variables of input TropCyclone to current. 
        Repeated events and centroids will be overwritten. Tracks are appended.
        
        Parameters:
            tc_haz (TropCyclone): TropCyclone instance to append to current

        Raises:
            ValueError
        """
        super(TropCyclone, self).append(tc_haz)
        self.track.extend(tc_haz.track)
        
    def track_info(self):
        """Get information about tracks"""
        # TODO infos track
        
    def plot_track(self):
        """Plot track """
        # TODO plot tracks
        
    def check(self):
        """Check if the attributes contain consistent data.

        Raises:
            ValueError
        """
        super(TropCyclone, self).check()
        check.size(len(self.event_id), self.track, 'track')

def read_ibtracs(file_name):
    """Read IBTrACS track file.

        Parameters:
            file_name (str): file name containing one IBTrACS track to read
        
        Returns:
            xarray.Dataset     
    """
    dfr = pd.read_csv(file_name)
    name = dfr['ibtracsID'].values[0]
    
    datetimes = list()
    for time in dfr['isotime'].values:
        year = np.fix(time/1e6)
        time = time - year*1e6
        month = np.fix(time/1e4)
        time = time - month*1e4
        day = np.fix(time/1e2)
        hour = time - day*1e2 
        datetimes.append(dt.datetime(int(year), int(month), int(day), \
                                     int(hour)))
    
    lat = dfr['cgps_lat'].values
    lon = dfr['cgps_lon'].values
    cen_pres = dfr['pcen'].values
    cen_pres_unit = 'kn'
    max_sus_wind = dfr['vmax'].values
    max_sus_wind_unit = 'kn'
    cen_pres = missing_pressure(cen_pres, max_sus_wind, lat, lon)
    category = set_category(max_sus_wind, max_sus_wind_unit)
    
    return xr.Dataset( \
        {'lon': ('time', lon), \
         'lat': ('time', lat), \
         'time_step': ('time', dfr['tint'].values), \
         'radius_max_wind': ('time', dfr['rmax'].values), \
         'max_sustained_wind': ('time', max_sus_wind), \
         'central_pressure': ('time', cen_pres), \
         'environmental_pressure': ('time', dfr['penv'].values)}, \
         coords={'time': datetimes}, \
         attrs={'max_sustained_wind_unit': max_sus_wind_unit, \
         'central_pressure_unit': cen_pres_unit, \
         'orig_event_flag': dfr['original_data'].values[0], \
         'name': name, \
         'data_provider': dfr['data_provider'].values[0], \
         'basin': dfr['gen_basin'].values[0], \
         'id_no': float(name.replace('N', '0').replace('S', '1')), \
         'category': category \
        })
    
def missing_pressure(cen_pres, v_max, lat, lon):
    """Deal with missing central pressures."""
    if np.argwhere(cen_pres < 0).size > 0:
        cen_pres = 1024.388 + 0.047*lat - 0.029*lon - 0.818*v_max
    return cen_pres

def set_category(max_sus_wind, max_sus_wind_unit):
    """Add storm category according to saffir-simpson hurricane scale
   -1 tropical depression
    0 tropical storm
    1 Hurrican category 1
    2 Hurrican category 2
    3 Hurrican category 3
    4 Hurrican category 4
    5 Hurrican category 5
    """
    ureg = UnitRegistry()
    if (max_sus_wind_unit == 'kn') or (max_sus_wind_unit == 'kt'):
        unit = ureg.knot
    elif max_sus_wind_unit == 'mph':
        unit = ureg.mile / ureg.hour
    elif max_sus_wind_unit == 'm/s':
        unit = ureg.meter / ureg.second
    elif max_sus_wind_unit == 'km/h':
        unit = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Wind not recorded in kn, conversion to kn needed.')
        raise ValueError
    max_wind_kn = (np.max(max_sus_wind) * unit).to(ureg.knot).magnitude
    
    return (np.argwhere(max_wind_kn < SAFFIR_SIM_CAT) - 1)[0][0]

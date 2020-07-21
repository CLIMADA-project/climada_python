"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---
Define Landslide class.
"""

__all__ = ['Landslide']

import logging
import os
import glob
import shlex
import subprocess
from scipy import sparse
from scipy.stats import binom
import geopandas
import pyproj
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import requests
import numpy as np
from haversine import haversine
from climada.hazard.base import Hazard
from climada.util.constants import DATA_DIR

LOGGER = logging.getLogger(__name__)

LS_FILE_DIR = os.path.join(DATA_DIR, 'system')

HAZ_TYPE = 'LS'


"""for future: implement a function that downloads COOLR data by command, not manually"""
# def get_coolr_shp(save_path=os.getcwd()):
#    """for LS_MODEL[0]: download most up-to-date version of historic LS records from
#    global landslide catalog (COOLR of NASA) in shape-file format (zip)"""

#   trials didn't work.
#    url = 'https://maps.nccs.nasa.gov/arcgis/home/item.html?id=ff4478ca84d24766bd79ac186bb60d9c#data'
#    resp_glc = requests.get(url=url)

#    url = 'https://data.nasa.gov/api/geospatial/h9d8-neg4?method=export&format=Shapefile'
#    # Timeout error, website is currently not working
#    LOGGER.info('requesting data from %s', url)
#    response = requests.get(url=url)
#    LOGGER.info('downloading content...')
#    open((save_path+'/global_LS_catalogue'+'.zip'), 'wb').write(response.content)
#

def get_nowcast_tiff(tif_type="monthly", starttime="", endtime="", save_path=os.getcwd()):
    """API request to get global monthly LS hazard map, averaged over 15 years from NASA
    or daily global LS nowcasting hazard map. Both from NASA.

    Paramters:
        tif_type (str): monthly or daily;
        starttime (str): 'yyyy-mm-dd' max. 90 days from now for daily LS tiff
        endtime(str): 'yyyy-mm-dd' for daily LS tiff
        save_path (str): save path for daily LS tiff (monthly gets downloaded into cwd)
    Returns:
        tiff files (monthly) to current_working_directory/
            [01-12]_ClimatologyMonthly_032818_9600x5400.tif or
        tiff files (daily) to save_path/LS_nowcast_date.tif
    """
    # the daily one is currently not producing any output
    if tif_type == "daily":
        if starttime > endtime:
            LOGGER.error("Start date must lie before end date. Please change")
            raise ValueError

        url = 'https://pmmpublisher.pps.eosdis.nasa.gov/opensearch'
        params = dict(
            q='global_landslide_nowcast',
            limit=50,
            startTime=starttime,
            endTime=endtime)

        resp = requests.get(url=url, params=params)
        data = resp.json()
        tif_url = []
        for item in data['items']:
            # extract json resonse snippet for tiff download
            tif_url.append(item['action'][1]['using'][2]['url'])

        resp_tif = []
        for url in tif_url:
            LOGGER.info('requesting %s', url)
            resp_tif.append(requests.get(url=url))
            LOGGER.info('downloading content...')
            with open((save_path + '/LS_nowcast_' + str(url[-12:-4]) + '.tif'), 'wb') as fp:
                fp.write(resp_tif[-1].content)

    elif tif_type == "monthly":

        command_line = ('curl -LO '
                        '"https://svs.gsfc.nasa.gov/vis/a000000/a004600/a004631/frames'
                        '/9600x5400_16x9_30p/MonthlyClimatology/'
                        '[01-12]_ClimatologyMonthly_032818_9600x5400.tif"')
        args = shlex.split(command_line)
        p = subprocess.Popen(args,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        p.communicate()


def combine_nowcast_tiff(ls_folder_path, search_criteria='LS*.tif', operator="maximum"):
    """Function to overlay several tiff files with landslide hazard data either by
    keeping maximum value per pixel or by summing up all pixel values.
    UPDATE: SOMETIMES WORKS, SOMETIMES NOT, ISSUE SEEMS TO BE WITH THE SHELL=TRUE COMMAND
    Parameters:
        ls_folder_path (str): folder path where landslide files are stored.
        search_criteria (str): common name identifier of files that should be combined.
            Either 'LS*.tif' for daily NASA nowcast maps or '*5400.tif' for monthly NASA maps.
        operator (str): "maximum" keeps maximum value for each pixel throughout files,
            "sum" sums up all values for each pixel throughout files.
     Returns:
        combined_nowcasts_LS.tif (tiff): 1 Tiff file combining all input tiffs.
    """


    # get names of all LS nowcast files present in LS folder
    ls_files = os.path.join(ls_folder_path, search_criteria)
    ls_files = glob.glob(ls_files)

    # WITH COMMAND LINE GDAL_CALC: keep maximum pixels when combining
    # loop over all LS nowcast files (inefficient)
    combined_layers_path = os.path.join(ls_folder_path, 'combined_nowcasts_LS.tif')
    if operator == "maximum":
        i = 0
        for file in ls_files:
            if i == 0:
                """
                Popen(['/usr/bin/env', 'progtorun', other, args], ...)
                /Users/evelynm/anaconda3/envs/climada_env_new/bin
                """
                command_line = 'gdal_calc.py --outfile=%s -A "%s" -B "%s" --calc="maximum(A,B)"' \
                % (combined_layers_path, file, file)
                args = shlex.split(command_line)
            else:
                command_line = 'gdal_calc.py --outfile=%s -A "%s" -B "%s" --calc="maximum(A,B)"'\
                % (combined_layers_path, combined_layers_path, file)
                args = shlex.split(command_line)
            i = i + 1
            p = subprocess.Popen(args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, shell=True)
            p.communicate()

    elif operator == "sum":
        i = 0
        for file in ls_files:
            if i == 0:
                command_line = 'gdal_calc.py --outfile=%s -A "%s" -B "%s" --calc="A+B"' \
                % (combined_layers_path, file, file)
                args = shlex.split(command_line)
            else:
                command_line = 'gdal_calc.py --outfile=%s -A "%s" -B "%s" --calc="A+B"' \
                % (combined_layers_path, combined_layers_path, file)
                args = shlex.split(command_line)
            i = i + 1
            p = subprocess.Popen(args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, shell=True)
            p.communicate()


class Landslide(Hazard):
    """Landslide Hazard set generation.
    Attributes:
    """

    def __init__(self):
        """Empty constructor."""
        Hazard.__init__(self, HAZ_TYPE)
        self.tag.haz_type = 'LS'


    def _get_window_from_coords(self, path_sourcefile, bbox=[]):
        ###### would fit better into base calss for sub-function of hazard.set_raster()########
        """get row, column, width and height required for rasterio window function
        from coordinate values of bounding box
        Parameters:
            bbox (array): [north, east, south, west]
            large_file (str): path of file from which window should be read in
        Returns:
            window_array (array): corner, width & height for Window() function of rasterio
        """
        with rasterio.open(path_sourcefile) as src:
            utm = pyproj.Proj(init='epsg:4326')  # Pass CRS of image from rasterio

        lonlat = pyproj.Proj(init='epsg:4326')
        lon, lat = (bbox[3], bbox[0])
        west, north = pyproj.transform(lonlat, utm, lon, lat)

        # What is the corresponding row and column in our image?
        row, col = src.index(west, north)  # spatial --> image coordinates

        lon, lat = (bbox[1], bbox[2])
        east, south = pyproj.transform(lonlat, utm, lon, lat)
        row2, col2 = src.index(east, south)
        width = abs(col2 - col)
        height = abs(row2 - row)

        window_array = [col, row, width, height]

        return window_array

    def _get_raster_meta(self, path_sourcefile, window_array):
        """get geo-meta data from raster files to set centroids adequately"""
        raster = rasterio.open(path_sourcefile, 'r',
                               window=Window(window_array[0], window_array[1],
                                             window_array[2], window_array[3]))
        pixel_width = raster.meta['transform'][0]
        pixel_height = raster.meta['transform'][4]

        return pixel_height, pixel_width

    def _intensity_cat_to_prob(self, max_prob):
        """convert NASA nowcasting categories into occurrence probabilities:
            highest value category value receives a prob of max_prob, lowest category value
            receives a prob value of 0"""
        self.intensity_cat = self.intensity.copy()  # save prob values
        self.intensity = self.intensity.astype(float)
        self.intensity.data = self.intensity.data.astype(float)
        max_value = float(max(self.intensity_cat.data))
        min_value = float(min(self.intensity_cat.data))

        for i, j in zip(*self.intensity.nonzero()):
            self.intensity[i, j] = float((self.intensity[i, j] - min_value) /
                                         (max_value - min_value) * max_prob)


    def _intensity_prob_to_binom(self, n_years):
        """convert occurrence probabilities in NGI/UNEP landslide hazard map into binary
        occurrences (yes/no) within a given time frame.

        Parameters
        ----------
        n_years : int
            the timespan of the probabilistic simulation in years

        Returns
        -------
        intensity_prob : csr matrix
            initial probabilities of ls occurrence per year per pixel
        intensity : csr matrix
            binary (0/1) occurrence within pixel
        """

        self.intensity_prob = self.intensity.copy()  # save prob values

        for i, j in zip(*self.intensity.nonzero()):
            if binom.rvs(n=n_years, p=self.intensity[i, j]) >= 1:
                self.intensity[i, j] = 1
            else:
                self.intensity[i, j] = 0

    def _intensity_binom_to_range(self, max_dist):
        """Affected neighbourhood' of pixels within certain threshold from ls occurrence
        can be included (takes long to compute, though).
        Parameters:
            max_dist (int): distance in metres (up to max ~1100) until which
                neighbouring pixels count as affected.
        Returns:
            intensity (csr matrix): range (0-1) where 0 = no occurrence, 1 = direct
                occurrence, ]0-1[ = relative distance to pixel with direct occurrence
        """
        self.intensity = self.intensity.tolil()
        # find all other pixels within certain distance from corresponding centroid,
        for i, j in zip(*self.intensity.nonzero()):
            subset_neighbours = self.centroids.geometry.cx[
                (self.centroids.coord[j][1] - 0.01):(self.centroids.coord[j][1] + 0.01),
                (self.centroids.coord[j][0] - 0.01):(self.centroids.coord[j][0] + 0.01)
            ]  # 0.01° = 1.11 km approximately
            for centroid in subset_neighbours:
                ix = subset_neighbours[subset_neighbours == centroid].index[0]
                # calculate dist, assign intensity [0-1] linearly until max_dist
                if haversine(self.centroids.coord[ix], self.centroids.coord[j], unit='m')\
                <= max_dist:
                    actual_dist = haversine(
                        self.centroids.coord[ix],
                        self.centroids.coord[j], unit='m')
                    # this step changes sparsity of matrix -->
                    # converted to lil_matrix, as more efficient
                    self.intensity[i, ix] = (max_dist - actual_dist) / max_dist
        self.intensity = self.intensity.tocsr()

    def plot_raw(self, ev_id=1, **kwargs):
        """Plot raw LHM data using imshow and without cartopy

        Parameters:
            ev_id (int, optional): event id. Default: 1.
            intensity (bool, optional): plot intensity if True, fraction otherwise
            kwargs (optional): arguments for imshow matplotlib function

        Returns:
            matplotlib.image.AxesImage
        """
        if not self.centroids.meta:
            LOGGER.error('No raster data set')
            raise ValueError
        try:
            event_pos = np.where(self.event_id == ev_id)[0][0]
        except IndexError:
            LOGGER.error('Wrong event id: %s.', ev_id)
            raise ValueError from IndexError

        return plt.imshow(self.intensity_prob[event_pos, :].toarray().
                          reshape(self.centroids.shape), **kwargs)

    def plot_events(self, ev_id=1, **kwargs):
        """Plot LHM event data using imshow and without cartopy

        Parameters:
            ev_id (int, optional): event id. Default: 1.
            intensity (bool, optional): plot intensity if True, fraction otherwise
            kwargs (optional): arguments for imshow matplotlib function

        Returns:
            matplotlib.image.AxesImage
        """
        if not self.centroids.meta:
            LOGGER.error('No raster data set')
            raise ValueError
        try:
            event_pos = np.where(self.event_id == ev_id)[0][0]
        except IndexError:
            LOGGER.error('Wrong event id: %s.', ev_id)
            raise ValueError from IndexError

        return plt.imshow(self.intensity[event_pos, :].toarray().
                          reshape(self.centroids.shape), **kwargs)

    def _get_hist_events(self, bbox, coolr_path):
        """for LS_MODEL[0]: load gdf with landslide event POINTS from
        global landslide catalog (COOLR of NASA) for bbox of interest"""
        ls_gdf = geopandas.read_file(coolr_path)
        ls_gdf_bbox = ls_gdf.cx[bbox[3]:bbox[1], bbox[2]:bbox[0]]
        return ls_gdf_bbox

    def set_ls_model_hist(self, bbox, path_sourcefile, check_plots=1):
        """set LS from historical records documented in the NASA COOLR initiative
        Parameters:
            bbox (array): [N, E , S, W] for which LS hazard should be calculated.
            path_sourcefile (str): path to shapefile with COOLR data, retrieved previously as
                described in tutorial
        Returns:
            Landslide() module: LS hazard set, historic
        """
        if not bbox:
            LOGGER.error('Empty bounding box, please set bounds.')
            raise ValueError()

        if not path_sourcefile:
            LOGGER.error('No sourcefile, please specify one containing historic LS points')
            raise ValueError()

        ls_gdf_bbox = self._get_hist_events(bbox, path_sourcefile)

        self.centroids.set_lat_lon(ls_gdf_bbox.latitude, ls_gdf_bbox.longitude)
        n_cen = ls_gdf_bbox.latitude.size  # number of centroids
        n_ev = n_cen
        self.intensity = sparse.csr_matrix(np.ones((n_ev, n_cen)))
        self.units = 'm/m'
        self.event_id = np.arange(n_ev, dtype=int)
        self.orig = np.zeros(n_ev, bool)
        self.frequency = np.ones(n_ev) / n_ev
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1)
        self.check()

        if check_plots == 1:
            self.centroids.plot()
        return self

    def set_ls_model_prob(self, bbox, ls_model="UNEP_NGI", path_sourcefile=[], n_years=500,
                          incl_neighbour=False, max_dist=1000, max_prob=0.000015, check_plots=1):
        """....
        Parameters:
            ls_model (str): UNEP_NGI (prob., UNEP/NGI) or NASA (prob., NASA Nowcast)
            bbox (array): [N, E , S, W] for which LS hazard should be calculated.
            n_years (int): timespan for probabilistic simulations. Default is 500y.
            incl_neighbour (bool): whether to include affected neighbouring pixels
                with dist <= max_dist. Default is false
            max_dist (int): distance until which neighbouring pixels should count as affected
                if incl_neighbour = True. Default is 1000m.
            max_prob (float): maximum occurence probability that should be assigned to
                categorical hazard maps (as in LS_MODEL[2]). Default is 0.000015
            path_sourcefile (str): if ls_model is UNEP_NGI, use  path to NGI/UNEP file,
                retrieved previously as descriped in tutorial and stored in climada/data.
                if ls_model is NASA  provide path to combined daily or
                monthly rasterfile, retrieved and aggregated
                previously with landslide.get_nowcast_tiff() and
                landslide.combine_nowcast_tiff().
        Returns:
            Landslide() module: probabilistic LS hazard set
        """

        if ls_model == "UNEP_NGI":
            path_sourcefile = os.path.join(LS_FILE_DIR, 'ls_pr_NGI_UNEP/ls_pr.tif')

            if not bbox:
                LOGGER.error('Empty bounding box, please set bounds.')
                raise ValueError()

            window_array = self._get_window_from_coords(path_sourcefile,
                                                        bbox)
            pixel_height, pixel_width = self._get_raster_meta(path_sourcefile, window_array)
            self.set_raster([path_sourcefile], window=Window(window_array[0], window_array[1],
                                                             window_array[3], window_array[2]))
            # prob values were initially multiplied by 1 mio
            self.intensity = self.intensity / 10e6
            self.centroids.set_raster_from_pix_bounds(bbox[0], bbox[3], pixel_height, pixel_width,
                                                      window_array[3], window_array[2])
            LOGGER.info('Generating landslides...')
            self._intensity_prob_to_binom(n_years)
            self.check()

            if incl_neighbour:
                LOGGER.info('Finding neighbouring pixels...')
                self.centroids.set_meta_to_lat_lon()
                self.centroids.set_geometry_points()
                self._intensity_binom_to_range(max_dist)
                self.check()

            if check_plots == 1:
                fig1 = plt.subplots(nrows=1, ncols=1)[0]
                self.plot_raw()
                fig1.suptitle('Raw data: Occurrence prob of LS per year', fontsize=14)

                fig2 = plt.subplots(nrows=1, ncols=1)[0]
                self.plot_events()
                fig2.suptitle('Prob. LS Hazard Set n_years = %i' % n_years, fontsize=14)

            return self

        elif ls_model == "NASA":
            if not bbox:
                LOGGER.error('Empty bounding box, please set bounds.')
                raise ValueError()

            if not path_sourcefile:
                LOGGER.error('Empty sourcefile, please specify')
                raise ValueError()
            window_array = self._get_window_from_coords(path_sourcefile, bbox)
            pixel_height, pixel_width = self._get_raster_meta(path_sourcefile, window_array)
            self.set_raster([path_sourcefile], window=Window(window_array[0], window_array[1],
                                                             window_array[3], window_array[2]))
            LOGGER.info('Setting probability values from categorical landslide hazard levels...')
            self._intensity_cat_to_prob(max_prob)
            self.centroids.set_raster_from_pix_bounds(bbox[0], bbox[3], pixel_height, pixel_width,
                                                      window_array[3], window_array[2])
            LOGGER.info('Generating binary landslides...')
            self._intensity_prob_to_binom(n_years)
            self.check()

            if incl_neighbour:
                LOGGER.info('Finding neighbouring pixels...')
                self.centroids.set_meta_to_lat_lon()
                self.centroids.set_geometry_points()
                self._intensity_binom_to_range(max_dist)
                self.check()

            if check_plots == 1:
                fig1, ax1 = plt.subplots(nrows=1, ncols=1)
                ax1 = self.plot_raw()
                fig1.suptitle('Raw data: Occurrence prob of LS per year', fontsize=14)

                fig2, ax2 = plt.subplots(nrows=1, ncols=1)
                ax2 = self.plot_events()
                fig2.suptitle('Prob. LS Hazard Set n_years = %i' % n_years, fontsize=14)

            return self

        else:
            LOGGER.error('Specify the LS model to be used for the hazard-set '
                         'generation as ls_model=str')
            raise KeyError

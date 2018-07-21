"""
Test BlackMarble base class.
"""
import unittest
import numpy as np
import scipy.sparse as sparse
import shapely
from cartopy.io import shapereader

from climada.entity.exposures.black_marble import country_iso_geom, BlackMarble, \
_process_land, _resample_land, _add_surroundings
from climada.entity.exposures.black_marble import MIN_LAT, MAX_LAT, MIN_LON, \
MAX_LON, NOAA_RESOLUTION_DEG

SHP_FN = shapereader.natural_earth(resolution='10m', \
    category='cultural', name='admin_0_countries')
SHP_FILE = shapereader.Reader(SHP_FN)

class TestCountryIso(unittest.TestCase):
    """Test country_iso function."""

    def test_che_kos_pass(self):
        """CHE"""
        country_name = ['Switzerland', 'Kosovo']
        iso_name = country_iso_geom(country_name, SHP_FILE)
        
        self.assertEqual(len(iso_name), len(country_name))
        self.assertTrue('CHE'in iso_name)
        self.assertTrue('KOS'in iso_name)
        self.assertEqual(iso_name['CHE'][0], 1)
        self.assertEqual(iso_name['CHE'][1], 'Switzerland')
        self.assertIsInstance(iso_name['CHE'][2], shapely.geometry.multipolygon.MultiPolygon)
        self.assertEqual(iso_name['KOS'][0], 2)
        self.assertEqual(iso_name['KOS'][1], 'Kosovo')
        self.assertIsInstance(iso_name['KOS'][2], shapely.geometry.multipolygon.MultiPolygon)

    def test_haiti_pass(self):
        """HTI"""
        country_name = ['HaITi']
        iso_name = country_iso_geom(country_name, SHP_FILE)
        
        self.assertEqual(len(iso_name), len(country_name))
        self.assertEqual(iso_name['HTI'][0], 1)
        self.assertEqual(iso_name['HTI'][1], 'Haiti')
        self.assertIsInstance(iso_name['HTI'][2], shapely.geometry.multipolygon.MultiPolygon)

    def test_wrong_fail(self):
        """Wrong name"""
        country_name = ['Kasovo']
        with self.assertRaises(ValueError):
            country_iso_geom(country_name, SHP_FILE)

    def test_bolivia_pass(self):
        """BOL"""
        country_name = ['Bolivia']
        iso_name = country_iso_geom(country_name, SHP_FILE)
        
        self.assertEqual(len(iso_name), len(country_name))
        self.assertEqual(iso_name['BOL'][0], 1)
        self.assertEqual(iso_name['BOL'][1], 'Bolivia')
        self.assertIsInstance(iso_name['BOL'][2], shapely.geometry.multipolygon.MultiPolygon)

    def test_korea_pass(self):
        """PRK"""
        country_name = ['Korea']
        with self.assertRaises(ValueError):
            country_iso_geom(country_name, SHP_FILE)

class TestNightLight(unittest.TestCase):
    """Test nightlight functions."""

    def test_process_land_brb_pass(self):
        """ Test _process_land function with fake Barbados."""
        country_iso = 'BRB'
        exp = BlackMarble()
        gdp = {country_iso: 4.225e9}
        income = {country_iso: 4}
        for cntry in list(SHP_FILE.records()):
            if cntry.attributes['ADM0_A3'] == country_iso:
                geom = cntry.geometry
        nightlight = sparse.lil.lil_matrix(np.zeros((16801, 43201)))
        nightlight[9000:9500, 14000:15000] = np.ones((500, 1000))
        nightlight[9350:9370, 14440:14450] = 0.4
        nightlight[9370:9400, 14450:14470] = 0.5
        nightlight = nightlight.tocsr()
        lat_nl = np.linspace(MIN_LAT + NOAA_RESOLUTION_DEG, MAX_LAT,
                             nightlight.shape[0])
        lon_nl = np.linspace(MIN_LON + NOAA_RESOLUTION_DEG, MAX_LON,
                             nightlight.shape[1])

        in_lat, in_lon, lat_mgrid, lon_mgrid, on_land = _process_land(exp, \
            geom, gdp[country_iso], income[country_iso], nightlight, \
            lat_nl, lon_nl)

        self.assertEqual(in_lat, (9366-1, 9400+1))
        self.assertEqual(in_lon, (14441-1, 14468+1))

        self.assertEqual(lat_mgrid.shape, (in_lat[1] - in_lat[0] + 1,
                         in_lon[1] - in_lon[0] + 1))
        self.assertEqual(lon_mgrid.shape, (in_lat[1] - in_lat[0] + 1,
                         in_lon[1] - in_lon[0] + 1))
        self.assertEqual(lat_mgrid[0][0], lat_nl[in_lat[0]])
        self.assertEqual(lat_mgrid[-1][0], lat_nl[in_lat[-1]])
        self.assertEqual(lon_mgrid[0][0], lon_nl[in_lon[0]])
        self.assertEqual(lon_mgrid[0][-1], lon_nl[in_lon[-1]])

        self.assertFalse(on_land[0][0])
        self.assertFalse(on_land[1][14])
        self.assertTrue(on_land[1][15])
        self.assertFalse(on_land[1][16])

        sum_nl = np.power(np.asarray(nightlight[in_lat[0]:in_lat[-1]+1, :] \
            [:, in_lon[0]:in_lon[-1]+1][on_land]).ravel(), 3).sum()
        self.assertEqual(exp.value[0],
            np.power(nightlight[in_lat[0]+1, in_lon[0]+15], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[1],
            np.power(nightlight[in_lat[0]+2, in_lon[0]+15], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[2],
            np.power(nightlight[in_lat[0]+2, in_lon[0]+16], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[3],
            np.power(nightlight[in_lat[0]+2, in_lon[0]+17], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[4],
            np.power(nightlight[in_lat[0]+2, in_lon[0]+18], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[0],
            np.power(nightlight[in_lat[0]+1, in_lon[0]+15], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[1],
            np.power(nightlight[in_lat[0]+2, in_lon[0]+15], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[2],
            np.power(nightlight[in_lat[0]+2, in_lon[0]+16], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[3],
            np.power(nightlight[in_lat[0]+2, in_lon[0]+17], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))
        self.assertEqual(exp.value[-1],
            np.power(nightlight[in_lat[1], in_lon[1]-17], 3)/sum_nl*gdp[country_iso]*(income[country_iso]+1))

        self.assertEqual(exp.coord[0, 0], lat_mgrid[on_land][0])
        self.assertEqual(exp.coord[0, 1], lon_mgrid[on_land][0])
        self.assertEqual(exp.coord[-1, 0], lat_mgrid[on_land][-1])
        self.assertEqual(exp.coord[-1, 1], lon_mgrid[on_land][-1])

    def test_resample_land_pass(self):
        """ Test _resample_land function with fake Barbados."""
        lat_nl = np.linspace(MIN_LAT + NOAA_RESOLUTION_DEG, MAX_LAT, 16801)
        lon_nl = np.linspace(MIN_LON + NOAA_RESOLUTION_DEG, MAX_LON, 43201)
        in_lat = (9365, 9401)
        in_lon = (14440, 14469)
        on_land = np.zeros((in_lat[1]-in_lat[0]+1, in_lon[1]-in_lon[0]+1)).astype(bool)
        on_land[10:15, 20:25] = True
        lat_mgrid, lon_mgrid = np.mgrid[
                lat_nl[in_lat[0]]:lat_nl[in_lat[1]]:complex(0, in_lat[1]-in_lat[0]+1),
                lon_nl[in_lon[0]]:lon_nl[in_lon[1]]:complex(0, in_lon[1]-in_lon[0]+1)]

        exp = BlackMarble()
        exp.value = np.arange(on_land.sum())
        exp.coord = np.empty((on_land.sum(), 2))
        exp.coord[:, 0] = lat_mgrid[on_land].ravel()
        exp.coord[:, 1] = lon_mgrid[on_land].ravel()
        res_km = 4.5
        ori_value = exp.value.copy()
        _resample_land(exp, res_km, lat_nl, lon_nl, in_lat, in_lon, on_land)

        self.assertEqual(exp.coord.shape, (2, 2))
        self.assertEqual(exp.coord[0][0], lat_mgrid[12, 20])
        self.assertEqual(exp.coord[1][0], lat_mgrid[12, 24])
        self.assertEqual(exp.coord[0][1], lon_mgrid[12, 20])
        self.assertAlmostEqual(exp.coord[1][1], lon_mgrid[12, 24], 13)

        ref_assigned = np.array([0, 0, 0, 1, 1,
                                 0, 0, 0, 1, 1,
                                 0, 0, 0, 1, 1,
                                 0, 0, 0, 1, 1,
                                 0, 0, 0, 1, 1])
        ref_val = np.empty((2,))
        ref_val[0] = float(ori_value[ref_assigned == 0].sum())
        ref_val[1] = float(ori_value[ref_assigned == 1].sum())
        self.assertTrue(np.array_equal(ref_val, exp.value))

    def test_add_surroundings(self):
        """ Test _add_surroundings function with fake Barbados."""
        lat_nl = np.linspace(MIN_LAT + NOAA_RESOLUTION_DEG, MAX_LAT, 16801)
        lon_nl = np.linspace(MIN_LON + NOAA_RESOLUTION_DEG, MAX_LON, 43201)
        in_lat = (9365, 9401)
        in_lon = (14440, 14469)
        on_land = np.zeros((in_lat[1]-in_lat[0]+1, in_lon[1]-in_lon[0]+1)).astype(bool)
        on_land[10:15, 20:25] = True
        lat_mgrid, lon_mgrid = np.mgrid[
                lat_nl[in_lat[0]]:lat_nl[in_lat[1]]:complex(0, in_lat[1]-in_lat[0]+1),
                lon_nl[in_lon[0]]:lon_nl[in_lon[1]]:complex(0, in_lon[1]-in_lon[0]+1)]

        exp = BlackMarble()
        exp.value = np.arange(on_land.sum())
        exp.coord = np.empty((on_land.sum(), 2))
        exp.coord[:, 0] = lat_mgrid[on_land].ravel()
        exp.coord[:, 1] = lon_mgrid[on_land].ravel()
        ori_value = exp.value.copy()
        _add_surroundings(exp, lat_mgrid, lon_mgrid, on_land)
        
        # every 50km
        surr_lat = lat_mgrid[np.logical_not(on_land)].ravel()[::50]
        surr_lon = lon_mgrid[np.logical_not(on_land)].ravel()[::50]
        
        self.assertEqual(exp.value.size, ori_value.size + surr_lat.size)
        self.assertTrue(np.array_equal(exp.value[-surr_lat.size:], np.zeros(surr_lat.size,)))

        self.assertEqual(exp.coord.shape[0], ori_value.size + surr_lat.size)
        self.assertTrue(np.array_equal(exp.coord.lat[-surr_lat.size:], surr_lat))
        self.assertTrue(np.array_equal(exp.coord.lon[-surr_lat.size:], surr_lon))

#    def test_pinto(self):
#        import matplotlib.pyplot as plt
#        plt.figure(figsize=(20,10))
#        nightlight, lat_nl, lon_nl, fn_nl = load_nightlight_noaa()
#        plt.imshow(np.flip(np.power(nightlight.todense(), 1),0))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCountryIso)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNightLight))
unittest.TextTestRunner(verbosity=2).run(TESTS)

"""
Test BlackMarble base class.
"""
import unittest
import numpy as np
import scipy.sparse as sparse
import shapely
from cartopy.io import shapereader

from climada.entity.exposures.black_marble import country_iso_geom, BlackMarble, \
_process_land, _add_surroundings, _get_gdp, _get_income_group, fill_econ_indicators, \
_set_econ_indicators
from climada.entity.exposures.black_marble import MIN_LAT, MAX_LAT, MIN_LON, \
MAX_LON, NOAA_RESOLUTION_DEG

SHP_FN = shapereader.natural_earth(resolution='10m', \
    category='cultural', name='admin_0_countries')
SHP_FILE = shapereader.Reader(SHP_FN)

class TestCountryIso(unittest.TestCase):
    """Test country_iso function."""

    def test_che_kos_pass(self):
        """CHE, KOS """
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

    def test_process_land_brb_1km_pass(self):
        """ Test _process_land function with fake Barbados."""
        country_iso = 'BRB'
        exp = BlackMarble()
        for cntry in list(SHP_FILE.records()):
            if cntry.attributes['ADM0_A3'] == country_iso:
                geom = cntry.geometry
        nightlight = sparse.lil.lil_matrix(np.ones((500, 1000)))
        nightlight[275:281, 333:334] = 0.4
        nightlight[275:281, 334:336] = 0.5
        nightlight = nightlight.tocsr()
        lat_nl = np.linspace(MIN_LAT + NOAA_RESOLUTION_DEG, MAX_LAT, 500)
        lon_nl = np.linspace(MIN_LON + NOAA_RESOLUTION_DEG, MAX_LON, 1000)

        lat_mgrid, lon_mgrid, on_land = _process_land(exp, \
            geom, nightlight, lat_nl, lon_nl, 1.0, 1.0)

        in_lat = (278, 280)
        in_lon = (333, 335)

        self.assertEqual(lat_mgrid.shape, (in_lat[1] - in_lat[0] + 1,
                         in_lon[1] - in_lon[0] + 1))
        self.assertEqual(lon_mgrid.shape, (in_lat[1] - in_lat[0] + 1,
                         in_lon[1] - in_lon[0] + 1))
        self.assertEqual(lat_mgrid[0][0], lat_nl[in_lat[0]])
        self.assertEqual(lat_mgrid[-1][0], lat_nl[in_lat[-1]])
        self.assertEqual(lon_mgrid[0][0], lon_nl[in_lon[0]])
        self.assertEqual(lon_mgrid[0][-1], lon_nl[in_lon[-1]])

        self.assertFalse(np.all(on_land[0][:]))
        self.assertFalse(np.all(on_land[2][:]))
        self.assertFalse(np.all(on_land[:][0]))
        self.assertFalse(np.all(on_land[:][2]))
        self.assertFalse(on_land[1][2])
        self.assertFalse(on_land[1][0])
        self.assertTrue(on_land[1][1])

        self.assertAlmostEqual(exp.value[0], nightlight[in_lat[0]+1, in_lon[0]+1])

        self.assertEqual(exp.coord[0, 0], lat_mgrid[on_land][0])
        self.assertEqual(exp.coord[0, 1], lon_mgrid[on_land][0])

    def test_process_land_brb_2km_pass(self):
        """ Test _process_land function with fake Barbados."""
        country_iso = 'BRB'
        exp = BlackMarble()
        for cntry in list(SHP_FILE.records()):
            if cntry.attributes['ADM0_A3'] == country_iso:
                geom = cntry.geometry
        nightlight = sparse.lil.lil_matrix(np.ones((500, 1000)))
        nightlight[275:281, 333:334] = 0.4
        nightlight[275:281, 334:336] = 0.5
        nightlight = nightlight.tocsr()
        lat_nl = np.linspace(MIN_LAT + NOAA_RESOLUTION_DEG, MAX_LAT, 500)
        lon_nl = np.linspace(MIN_LON + NOAA_RESOLUTION_DEG, MAX_LON, 1000)

        res_fact = 2.0
        lat_mgrid, lon_mgrid, on_land = _process_land(exp, \
            geom, nightlight, lat_nl, lon_nl,res_fact, 0.5)

        in_lat = (278, 280)
        in_lon = (333, 335)

        self.assertEqual(lat_mgrid.shape, (res_fact*(in_lat[1] - in_lat[0] + 1),
                         res_fact*(in_lon[1] - in_lon[0] + 1)))
        self.assertEqual(lon_mgrid.shape, (res_fact*(in_lat[1] - in_lat[0] + 1),
                         res_fact*(in_lon[1] - in_lon[0] + 1)))
        self.assertEqual(lat_mgrid[0][0], lat_nl[in_lat[0]])
        self.assertEqual(lat_mgrid[-1][0], lat_nl[in_lat[-1]])
        self.assertEqual(lon_mgrid[0][0], lon_nl[in_lon[0]])
        self.assertEqual(lon_mgrid[0][-1], lon_nl[in_lon[-1]])

        self.assertFalse(np.all(on_land[0][:]))
        self.assertFalse(np.all(on_land[3][:]))
        self.assertFalse(np.all(on_land[4][:]))
        self.assertFalse(np.all(on_land[5][:]))
        
        self.assertFalse(np.all(on_land[:][0]))
        self.assertFalse(np.all(on_land[:][1]))
        self.assertFalse(np.all(on_land[:][2]))
        self.assertFalse(np.all(on_land[:][4]))
        self.assertFalse(np.all(on_land[:][5]))
        self.assertTrue(on_land[1][3])
        self.assertTrue(on_land[2][3])
        
        self.assertAlmostEqual(exp.value[0], 0.5096)
        self.assertAlmostEqual(exp.value[1], 0.5096)

        self.assertEqual(exp.coord[0, 0], lat_mgrid[on_land][0])
        self.assertEqual(exp.coord[0, 1], lon_mgrid[on_land][0])

        self.assertEqual(exp.coord[1, 0], lat_mgrid[on_land][1])
        self.assertEqual(exp.coord[1, 1], lon_mgrid[on_land][1])
        
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

class TestEconIndices(unittest.TestCase):
    """Test functions to get economic indices."""
    
    def test_income_grp_aia_pass(self):
        """ Test _get_income_group function Anguilla."""
        cntry_info = {'AIA': [1, 'Anguilla', 'geom']}
        ref_year = 2012
        with self.assertLogs('climada.entity.exposures', level='INFO') as cm:
            _get_income_group(cntry_info, ref_year, SHP_FILE)
            
        cntry_info_ref = {'AIA': [1, 'Anguilla', 'geom', 3]}
        self.assertIn('Income group AIA: 3', cm.output[0])
        self.assertEqual(cntry_info, cntry_info_ref)

    def test_income_grp_sxm_2012_pass(self):
        """ Test _get_income_group function Sint Maarten."""
        cntry_info = {'SXM': [1, 'Sint Maarten', 'geom']}
        ref_year = 2012
        with self.assertLogs('climada.entity.exposures', level='INFO') as cm:
            _get_income_group(cntry_info, ref_year, SHP_FILE)
        
        cntry_info_ref = {'SXM': [1, 'Sint Maarten', 'geom', 4]}
        self.assertIn('Income group SXM 2012: 4.', cm.output[0])
        self.assertEqual(cntry_info, cntry_info_ref)

    def test_income_grp_sxm_1999_pass(self):
        """ Test _get_income_group function Sint Maarten."""
        cntry_info = {'SXM': [1, 'Sint Maarten', 'geom']}
        ref_year = 1999
        with self.assertLogs('climada.entity.exposures', level='INFO') as cm:
            _get_income_group(cntry_info, ref_year, SHP_FILE)
            
        cntry_info_ref = {'SXM': [1, 'Sint Maarten', 'geom', 4]}
        self.assertIn('Income group SXM 2010: 4.', cm.output[0])
        self.assertEqual(cntry_info, cntry_info_ref)

    def test_get_gdp_aia_2012_pass(self):
        """ Test _get_gdp function Anguilla."""
        cntry_info = {'AIA': [1, 'Anguilla', 'geom']}
        ref_year = 2012
        with self.assertLogs('climada.entity.exposures', level='INFO') as cm:
            _get_gdp(cntry_info, ref_year, SHP_FILE)
            
        cntry_info_ref = {'AIA': [1, 'Anguilla', 'geom', 1.754e+08]}
        self.assertIn('GDP AIA 2009: 1.754e+08', cm.output[0])
        self.assertEqual(cntry_info, cntry_info_ref)

    def test_get_gdp_sxm_2012_pass(self):
        """ Test _get_gdp function Sint Maarten."""
        cntry_info = {'SXM': [1, 'Sint Maarten', 'geom']}
        ref_year = 2012
        with self.assertLogs('climada.entity.exposures', level='INFO') as cm:
            _get_gdp(cntry_info, ref_year, SHP_FILE)
        
        cntry_info_ref = {'SXM': [1, 'Sint Maarten', 'geom', 3.658e+08]}
        self.assertIn('GDP SXM 2014: 3.658e+08', cm.output[0])
        self.assertEqual(cntry_info, cntry_info_ref)

    def test_get_gdp_esp_1950_pass(self):
        """ Test _get_gdp function Sint Maarten."""
        cntry_info = {'ESP': [1, 'Spain', 'geom']}
        ref_year = 1950
        with self.assertLogs('climada.entity.exposures', level='INFO') as cm:
            _get_gdp(cntry_info, ref_year, SHP_FILE)
            
        cntry_info_ref = {'ESP': [1, 'Spain', 'geom', 12072126075.397]}
        self.assertIn('GDP ESP 1960: 1.207e+10', cm.output[0])
        self.assertEqual(cntry_info, cntry_info_ref)
        
    def test_fill_econ_indicators_pass(self):
        ref_year = 2015
        country_isos = {'CHE': [1, 'Switzerland', 'che_geom'],
                        'ZMB': [2, 'Zambia', 'zmb_geom']
                       }
        fill_econ_indicators(ref_year, country_isos, SHP_FILE)
        country_isos_ref = {'CHE': [1, 'Switzerland', 'che_geom', 2015, 679289166858.236, 4],
                            'ZMB': [2, 'Zambia', 'zmb_geom', 2015, 21154394545.895, 2]
                           }
        self.assertEqual(country_isos, country_isos_ref)

    def test_fill_econ_indicators_kwargs_pass(self):
        ref_year = 2015
        country_isos = {'CHE': [1, 'Switzerland', 'che_geom'],
                        'ZMB': [2, 'Zambia', 'zmb_geom']
                       }
        gdp = {'CHE': 1.2, 'ZMB': 1.3}
        inc_grp = {'CHE': 3, 'ZMB': 4}
        kwargs = {'gdp': gdp, 'inc_grp': inc_grp}
        fill_econ_indicators(ref_year, country_isos, SHP_FILE, **kwargs)
        country_isos_ref = {'CHE': [1, 'Switzerland', 'che_geom', 2015, gdp['CHE'], inc_grp['CHE']],
                            'ZMB': [2, 'Zambia', 'zmb_geom', 2015, gdp['ZMB'], inc_grp['ZMB']]
                           }
        self.assertEqual(country_isos, country_isos_ref)

    def test_set_econ_indicators_pass(self):
        """ Test _set_econ_indicators pass."""
        exp = BlackMarble()
        exp.value = np.arange(0, 20, 0.1)
        gdp = 4.225e9
        inc_grp = 4
        _set_econ_indicators(exp, gdp, inc_grp)
        
        self.assertAlmostEqual(exp.value.sum(), gdp*(inc_grp+1), 5)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEconIndices)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCountryIso))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNightLight))
unittest.TextTestRunner(verbosity=2).run(TESTS)

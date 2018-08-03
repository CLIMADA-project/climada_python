"""
Test BlackMarble base class.
"""
import unittest
import numpy as np
import scipy.sparse as sparse
import shapely
from cartopy.io import shapereader
from sklearn.neighbors import DistanceMetric

from climada.entity.exposures.black_marble import country_iso_geom, BlackMarble, \
_process_land, _get_gdp, _get_income_group, fill_econ_indicators, add_sea, \
_set_econ_indicators, _fill_admin1_geom, _filter_admin1
from climada.entity.exposures.nightlight import NOAA_BORDER, NOAA_RESOLUTION_DEG
from climada.util.constants import ONE_LAT_KM
from climada.util.coordinates import coord_on_land

SHP_FN = shapereader.natural_earth(resolution='10m', \
    category='cultural', name='admin_0_countries')
SHP_FILE = shapereader.Reader(SHP_FN)

ADM1_FILE = shapereader.natural_earth(resolution='10m',
                                      category='cultural',
                                      name='admin_1_states_provinces')
ADM1_FILE = shapereader.Reader(ADM1_FILE)

class TestCountryIso(unittest.TestCase):
    """Test country_iso function."""

    def test_che_kos_pass(self):
        """CHE, KOS """
        country_name = ['Switzerland', 'Kosovo']
        iso_name, _ = country_iso_geom(country_name, SHP_FILE)
        
        self.assertEqual(len(iso_name), len(country_name))
        self.assertTrue('CHE'in iso_name)
        self.assertTrue('KOS'in iso_name)
        self.assertEqual(iso_name['CHE'][0], 41)
        self.assertEqual(iso_name['CHE'][1], 'Switzerland')
        self.assertIsInstance(iso_name['CHE'][2], shapely.geometry.multipolygon.MultiPolygon)
        self.assertEqual(iso_name['KOS'][0], 252)
        self.assertEqual(iso_name['KOS'][1], 'Kosovo')
        self.assertIsInstance(iso_name['KOS'][2], shapely.geometry.multipolygon.MultiPolygon)

    def test_haiti_pass(self):
        """HTI"""
        country_name = ['HaITi']
        iso_name, _ = country_iso_geom(country_name, SHP_FILE)
        
        self.assertEqual(len(iso_name), len(country_name))
        self.assertEqual(iso_name['HTI'][0], 100)
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
        iso_name, _ = country_iso_geom(country_name, SHP_FILE)
        
        self.assertEqual(len(iso_name), len(country_name))
        self.assertEqual(iso_name['BOL'][0], 31)
        self.assertEqual(iso_name['BOL'][1], 'Bolivia')
        self.assertIsInstance(iso_name['BOL'][2], shapely.geometry.multipolygon.MultiPolygon)

    def test_korea_pass(self):
        """PRK"""
        country_name = ['Korea']
        with self.assertRaises(ValueError):
            country_iso_geom(country_name, SHP_FILE)

class TestProvinces(unittest.TestCase):
    """Tst black marble with admin1."""
    def test_fill_admin1_geom_pass(self):
        """Test function _fill_admin1_geom pass."""
        iso3 = 'ESP'
        admin1_rec = list(ADM1_FILE.records())

        prov_list = ['Barcelona']
        res_bcn = _fill_admin1_geom(iso3, admin1_rec, prov_list)
        self.assertEqual(len(res_bcn), 1)
        self.assertIsInstance(res_bcn[0], shapely.geometry.multipolygon.MultiPolygon)
        
        prov_list = ['Barcelona', 'Tarragona']
        res_bcn = _fill_admin1_geom(iso3, admin1_rec, prov_list)
        self.assertEqual(len(res_bcn), 2)
        self.assertIsInstance(res_bcn[0], shapely.geometry.multipolygon.MultiPolygon)
        self.assertIsInstance(res_bcn[1], shapely.geometry.multipolygon.MultiPolygon)

    def test_fill_admin1_geom_fail(self):
        """Test function _fill_admin1_geom fail."""
        iso3 = 'CHE'
        admin1_rec = list(ADM1_FILE.records())

        prov_list = ['Barcelona']
        with self.assertRaises(ValueError):
            with self.assertLogs('climada.entity.exposures.black_marble', level='ERROR') as cm:
                _fill_admin1_geom(iso3, admin1_rec, prov_list)
        self.assertIn('Barcelona not found. Possible provinces of CHE are: ', cm.output[0])
        
    def test_country_iso_geom_pass(self):
        """Test country_iso_geom pass."""
        countries = ['Switzerland']
        _, cntry_admin1 = country_iso_geom(countries, SHP_FILE)
        self.assertEqual(cntry_admin1, {'CHE': []})
        self.assertIsInstance(countries, list)

        countries = {'Switzerland': ['Zürich']}
        _, cntry_admin1 = country_iso_geom(countries, SHP_FILE)
        self.assertEqual(len(cntry_admin1['CHE']), 1)
        self.assertIsInstance(cntry_admin1['CHE'][0], shapely.geometry.multipolygon.MultiPolygon)

    def test_filter_admin1_pass(self):
        """Test _filter_admin1 pass."""
        exp_bkmrb = BlackMarble()
        exp_bkmrb.value = np.arange(100)
        exp_bkmrb.coord = np.empty((100, 2))
        exp_bkmrb.coord[:, 0] = 41.39
        exp_bkmrb.coord[:, 1] = np.linspace(0, 3, 100)
        admin1_rec = list(ADM1_FILE.records())
        for rec in admin1_rec:
            if 'Barcelona' in rec.attributes['name']:
                bcn_geom = rec.geometry

        _filter_admin1(exp_bkmrb, bcn_geom)
        for coord in exp_bkmrb.coord:
            self.assertTrue(bcn_geom.contains(shapely.geometry.Point([coord[1],coord[0]])))

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

        coord_nl = np.empty((2, 2))
        coord_nl[0, :] = [NOAA_BORDER[1]+NOAA_RESOLUTION_DEG,
                          0.2805444221776838]
        coord_nl[1, :] = [NOAA_BORDER[0]+NOAA_RESOLUTION_DEG,
                          0.3603520186853473]

        res_fact = 1.0
        res_km = 1.0
        _process_land(exp, geom, nightlight, coord_nl, res_fact, res_km)

        lat_mgrid = np.array([[12.9996827 , 12.9996827 , 12.9996827 ],
                              [13.28022712, 13.28022712, 13.28022712],
                              [13.56077154, 13.56077154, 13.56077154]])

        lon_mgrid = np.array([[-59.99444444, -59.63409243, -59.27374041],
                              [-59.99444444, -59.63409243, -59.27374041],
                              [-59.99444444, -59.63409243, -59.27374041]])
    
        on_land = np.array([[False, False, False],
                            [False,  True, False],
                            [False, False, False]])
    
        in_lat = (278, 280)
        in_lon = (333, 335)
        self.assertAlmostEqual(exp.value[0], nightlight[in_lat[0]+1, in_lon[0]+1])
        self.assertAlmostEqual(exp.coord[0, 0], lat_mgrid[on_land][0])
        self.assertAlmostEqual(exp.coord[0, 1], lon_mgrid[on_land][0])
        
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

        coord_nl = np.empty((2, 2))
        coord_nl[0, :] = [NOAA_BORDER[1]+NOAA_RESOLUTION_DEG,
                          0.2805444221776838]
        coord_nl[1, :] = [NOAA_BORDER[0]+NOAA_RESOLUTION_DEG,
                          0.3603520186853473]

        res_fact = 2.0
        res_km = 0.5
        _process_land(exp, geom, nightlight, coord_nl, res_fact, res_km)

        lat_mgrid = np.array([[12.9996827, 12.9996827, 12.9996827, 12.9996827, 12.9996827, 12.9996827 ],
                              [13.11190047, 13.11190047, 13.11190047, 13.11190047, 13.11190047, 13.11190047],
                              [13.22411824, 13.22411824, 13.22411824, 13.22411824, 13.22411824, 13.22411824],
                              [13.33633601, 13.33633601, 13.33633601, 13.33633601, 13.33633601, 13.33633601],
                              [13.44855377, 13.44855377, 13.44855377, 13.44855377, 13.44855377, 13.44855377],
                              [13.56077154, 13.56077154, 13.56077154, 13.56077154, 13.56077154, 13.56077154]])

        lon_mgrid = np.array([[-59.99444444, -59.85030364, -59.70616283, -59.56202202, -59.41788121, -59.27374041],
                              [-59.99444444, -59.85030364, -59.70616283, -59.56202202, -59.41788121, -59.27374041],
                              [-59.99444444, -59.85030364, -59.70616283, -59.56202202, -59.41788121, -59.27374041],
                              [-59.99444444, -59.85030364, -59.70616283, -59.56202202, -59.41788121, -59.27374041],
                              [-59.99444444, -59.85030364, -59.70616283, -59.56202202, -59.41788121, -59.27374041],
                              [-59.99444444, -59.85030364, -59.70616283, -59.56202202, -59.41788121, -59.27374041]])
    
        on_land = np.array([[False, False, False, False, False, False],
                            [False, False, False,  True, False, False],
                            [False, False, False,  True, False, False],
                            [False, False, False, False, False, False],
                            [False, False, False, False, False, False],
                            [False, False, False, False, False, False]])

        self.assertAlmostEqual(exp.value[0], 0.5096)
        self.assertAlmostEqual(exp.value[1], 0.5096)

        self.assertAlmostEqual(exp.coord[0, 0], lat_mgrid[on_land][0])
        self.assertAlmostEqual(exp.coord[0, 1], lon_mgrid[on_land][0])

        self.assertAlmostEqual(exp.coord[1, 0], lat_mgrid[on_land][1])
        self.assertAlmostEqual(exp.coord[1, 1], lon_mgrid[on_land][1])

    def test_add_sea_pass(self):
        """Test add_sea function with fake data."""
        exp = BlackMarble()
        
        exp.value = np.arange(0, 1.0e6, 1.0e5)
        
        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12
        exp.coord = np.zeros((10, 2))
        exp.coord[:, 0] = np.linspace(min_lat, max_lat, 10)
        exp.coord[:, 1] = np.linspace(min_lon, max_lon, 10)
        exp.region_id = np.ones(10)
        
        sea_coast = 100
        sea_res_km = 50
        sea_res = (sea_coast, sea_res_km)
        add_sea(exp, sea_res)
        exp.check()
       
        sea_coast /= ONE_LAT_KM
        sea_res_km /= ONE_LAT_KM
        
        min_lat = min_lat - sea_coast
        max_lat = max_lat + sea_coast
        min_lon = min_lon - sea_coast
        max_lon = max_lon + sea_coast
        self.assertEqual(np.min(exp.coord.lat), min_lat)
        self.assertEqual(np.min(exp.coord.lon), min_lon)
        self.assertTrue(np.array_equal(exp.value[:10], np.arange(0, 1.0e6, 1.0e5)))
        
        on_sea_lat = exp.coord[11:, 0]
        on_sea_lon = exp.coord[11:, 1]
        res_on_sea = coord_on_land(on_sea_lat, on_sea_lon)
        res_on_sea = np.logical_not(res_on_sea)
        self.assertTrue(np.all(res_on_sea))
        
        dist = DistanceMetric.get_metric('haversine')
        self.assertAlmostEqual(dist.pairwise([[exp.coord[-1][1], exp.coord[-1][0]],
            [exp.coord[-2][1], exp.coord[-2][0]]])[0][1], sea_res_km)

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
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProvinces))
unittest.TextTestRunner(verbosity=2).run(TESTS)

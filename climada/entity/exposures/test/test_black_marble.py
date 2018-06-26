"""
Test BlackMarble base class.
"""
import unittest

from climada.entity.exposures.black_marble import country_iso, untar_stable_nightlight, load_nightlight

class TestCountryIso(unittest.TestCase):
    """Test country_iso function."""
    def test_germany_pass(self):
        """DEU """
        country_name = 'Germany'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'DEU')

    def test_switzerland_pass(self):
        """CHE"""
        country_name = 'Switzerland'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'CHE')

    def test_haiti_pass(self):
        """HTI"""
        country_name = 'Haiti'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'HTI')

    def test_barbados_pass(self):
        """BRB"""
        country_name = 'Barbados'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'BRB')

    def test_zambia_pass(self):
        """ZMB"""
        country_name = 'Zambia'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'ZMB')        
    
    def test_pass(self):
#        file_tar = '/Users/aznarsig/Documents/Python/climada_python/climada/entity/exposures/test/F182012.v4.tar'
#        res = untar_stable_nightlight(file_tar)
#        print('hecho', res)
        nightlight, lat, lon, fn_light = load_nightlight(2007, 'F15')
#        print('fin', fn)
        print(nightlight.shape)
        print(lat.size)
        print(lon.size)
        print(fn_light)
#        print('fin')
#        import os
#        import gzip
#        import shutil
#        import matplotlib.pyplot as plt
#        file = '/Users/aznarsig/Documents/Python/climada_python/climada/test/data/F182012.v4c_web.stable_lights.avg_vis.tif.gz'
##        f_in = gzip.open(fn_light, 'rb')
##        nightlight = plt.imread(f_in)
##        print(type(nightlight))
##        f_in.close()
#        
#        #    f_tif = os.path.splitext(extract_name)[0]
#        with gzip.open(file, 'rb') as f_in:
#            with open('/Users/aznarsig/Documents/Python/climada_python/climada/test/data/F182012.v4c_web.stable_lights.avg_vis.tif', 'wb') as f_out:
#                shutil.copyfileobj(f_in, f_out)
#        nightlight = plt.imread('/Users/aznarsig/Documents/Python/climada_python/climada/test/data/F182012.v4c_web.stable_lights.avg_vis.tif')
#        os.remove('/Users/aznarsig/Documents/Python/climada_python/climada/test/data/F182012.v4c_web.stable_lights.avg_vis.tif')
##        file = '/Users/aznarsig/Desktop/MS_San_Salvador_event_2520.tif.gz'
##        with gzip.open(file, 'rb') as f_in:
##            nightlight = plt.imread(f_in)
#        print('fin', type(nightlight))
#        print('size', nightlight.shape)
##        url = 'https://ngdc.noaa.gov/eog/data/web_data/v4composites/F182012.v4.tar'
##        nightlight = plt.imread(url)
##        print(type(nightlight))
#        
#        
##        with gzip.open(fn_light, 'rb') as f_in:
##            nightlight = plt.imread(f_in)
##        print('acabe', type(nightlight))
##        try_url = 'https://ngdc.noaa.gov/eog/data/web_data/v4composites/F172010.v4.tar'
##        try:
##         file = download_file(try_url)
##        except ValueError:
##            pass
##        if 'file' in locals():
##            print('bien1')
##        else:
##            print('bien2')
##        import glob
##        from climada import DATA_DIR
###        print('aqui', DATA_DIR)
###        print('aqui2', os.path.join(os.path.abspath(DATA_DIR), '*2013*v4c_web.stable_lights.avg_vis.tif'))
##        file_name = os.path.join(os.path.abspath(DATA_DIR), '*2013*v4c_web.stable_lights.avg_vis.tif')
##        if glob.glob(file_name):
##            print('siiiiiiii', file_name)
##            print(glob.glob(file_name)[0])
##        else:
##            print('noooooooooo', file_name)
##            print(glob.glob(file_name))
        
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCountryIso)
#TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAssign))
unittest.TextTestRunner(verbosity=2).run(TESTS)

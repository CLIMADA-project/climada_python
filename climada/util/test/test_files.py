"""
Test files_handler module.
"""

import os
import unittest

from climada.util.files_handler import to_list, get_file_names, download_file
from climada.util.constants import DATA_DIR

class TestDownloadUrl(unittest.TestCase):
    """Test download_file function """
    def test_wrong_url_fail(self):
        """Error raised when wrong url."""
        url = 'https://ngdc.noaa.gov/eog/data/web_data/v4composites/F172012.v4.tar'
        with self.assertRaises(ValueError):
            download_file(url)

class TestToStrList(unittest.TestCase):
    """Test to_list function"""
    def test_identity_pass(self):
        """Returns the same list if its length is correct."""
        num_exp = 3
        values = ['hi', 'ho', 'ha']
        val_name = 'values'
        out = to_list(num_exp, values, val_name)
        self.assertEqual(values, out)
        
    def test_one_to_list(self):
        """When input is a string or list with one element, it returns a list 
        with the expected number of elments repeated"""
        num_exp = 3
        values = 'hi'
        val_name = 'values'
        out = to_list(num_exp, values, val_name)
        self.assertEqual(['hi', 'hi', 'hi'], out)
        
        values = ['ha']
        out = to_list(num_exp, values, val_name)
        self.assertEqual(['ha', 'ha', 'ha'], out)

    def test_list_wrong_length_fail(self):
        """When input is list of neither expected size nor one, fail."""
        num_exp = 3
        values = ['1', '2']
        val_name = 'values'
        
        with self.assertLogs('climada.util.files_handler', level='ERROR') as cm:
            to_list(num_exp, values, val_name)
        self.assertIn("Provide one or 3 values.", cm.output[0])
        
class TestGetFileNames(unittest.TestCase):
    """Test get_file_names function"""
    def test_one_file_copy(self):
        """If input is one file name, return a list with this file name"""
        file_name = "test.mat"
        out = get_file_names(file_name)
        self.assertEqual([file_name], out)

    def test_several_file_copy(self):
        """If input is a list with several file names, return the same list"""
        file_name = ["test1.mat", "test2.mat", "test3.mat", "test4.mat"]
        out = get_file_names(file_name)
        self.assertEqual(file_name, out)

    def test_folder_contents(self):
        """If input is one folder name, return a list with containg files. 
        Folder names are not contained."""
        file_name = os.path.join(DATA_DIR, 'demo')
        out = get_file_names(file_name)
        for file in out:
            self.assertEqual('.', os.path.splitext(file)[1][0])
        
        file_name = DATA_DIR
        out = get_file_names(file_name)
        for file in out:
            self.assertNotEqual('', os.path.splitext(file)[1])

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestToStrList)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGetFileNames))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDownloadUrl))
unittest.TextTestRunner(verbosity=2).run(TESTS)

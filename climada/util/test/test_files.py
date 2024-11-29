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

Test files_handler module.
"""

import shutil
import tempfile
import time
import unittest
from concurrent.futures import thread
from pathlib import Path

from climada.util.constants import DEMO_DIR, ENT_TEMPLATE_XLS, GLB_CENTROIDS_MAT
from climada.util.files_handler import (
    Download,
    Downloader,
    download_file,
    get_file_names,
    to_list,
)


class TestDownloader(unittest.TestCase):
    """Test Downloader methods"""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        time.sleep(1)
        shutil.rmtree(cls.tmpdir)
        return super().tearDownClass()

    def setUp(self):
        self.downloader = Downloader(Path(self.tmpdir, ".downloads.db"))

    def tearDown(self):
        self.downloader.DB.close()

    def test_failing_download(self):
        Downloader.MAX_WAITING_PERIOD = 0.01
        with self.assertRaises(Download.Failed) as dfe:
            self.downloader.download(
                url="http://where/ev.er", target_dir=self.tmpdir, file_name="who cares"
            )


class TestDownloadUrl(unittest.TestCase):
    """Test download_file function"""

    def test_wrong_url_fail(self):
        """Error raised when wrong url."""
        url = "https://ngdc.noaa.gov/eog/data/web_data/v4composites/F172012.v4.tar"
        try:
            with self.assertRaises(ValueError):
                download_file(url)
        except IOError:
            pass


class TestToStrList(unittest.TestCase):
    """Test to_list function"""

    def test_identity_pass(self):
        """Returns the same list if its length is correct."""
        num_exp = 3
        values = ["hi", "ho", "ha"]
        val_name = "values"
        out = to_list(num_exp, values, val_name)
        self.assertEqual(values, out)

    def test_one_to_list(self):
        """When input is a string or list with one element, it returns a list
        with the expected number of elments repeated"""
        num_exp = 3
        values = "hi"
        val_name = "values"
        out = to_list(num_exp, values, val_name)
        self.assertEqual(["hi", "hi", "hi"], out)

        values = ["ha"]
        out = to_list(num_exp, values, val_name)
        self.assertEqual(["ha", "ha", "ha"], out)

    def test_list_wrong_length_fail(self):
        """When input is list of neither expected size nor one, fail."""
        num_exp = 3
        values = ["1", "2"]
        val_name = "values"

        with self.assertRaises(ValueError) as cm:
            to_list(num_exp, values, val_name)
        self.assertIn("Provide one or 3 values.", str(cm.exception))


class TestGetFileNames(unittest.TestCase):
    """Test get_file_names function. Only works with actually existing
    files and directories."""

    def test_one_file_copy(self):
        """If input is one file name, return a list with this file name"""
        file_name = GLB_CENTROIDS_MAT
        out = get_file_names(file_name)
        self.assertEqual([str(file_name)], out)

    def test_several_file_copy(self):
        """If input is a list with several file names, return the same list"""
        file_name = [GLB_CENTROIDS_MAT, ENT_TEMPLATE_XLS]
        out = get_file_names(file_name)
        self.assertEqual([str(x) for x in file_name], out)

    def test_folder_contents(self):
        """If input is one folder name, return a list with containg files.
        Folder names are not contained."""
        file_name = DEMO_DIR
        out = get_file_names(file_name)
        self.assertGreater(len(out), 0)
        for file in out:
            self.assertTrue(Path(file).is_file())

    def test_wrong_argument(self):
        """If the input contains a non-existing file, an empyt directory or a pattern that is not
        matched, the method should raise a ValueError."""
        empty_dir = DEMO_DIR.parent
        with self.assertRaises(ValueError) as ve:
            get_file_names(str(empty_dir))
        self.assertIn("no files", str(ve.exception))

        no_file = "this is not a file"
        with self.assertRaises(ValueError) as ve:
            get_file_names(no_file)
        self.assertIn("cannot find", str(ve.exception))

    def test_globbing(self):
        """If input is a glob pattern, return a list of matching visible
        files; omit folders.
        """
        file_name = DEMO_DIR
        out = get_file_names(f"{file_name}/*")

        tmp_files = [
            str(f)
            for f in Path(file_name).iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

        self.assertListEqual(sorted(tmp_files), sorted(out))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestToStrList)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGetFileNames))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDownloadUrl))
    unittest.TextTestRunner(verbosity=2).run(TESTS)

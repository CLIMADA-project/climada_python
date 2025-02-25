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

from climada.hazard.io import LOGGER
from climada.util.constants import DEMO_DIR, ENT_TEMPLATE_XLS, GLB_CENTROIDS_MAT
from climada.util.files_handler import (
    Downloader,
    DownloadFailed,
    download_file,
    file_checksum,
    get_file_names,
    to_list,
)


class TestDownloader(unittest.TestCase):
    """Test Downloader methods"""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

    def setUp(self):
        Downloader.MAX_WAITING_PERIOD = 0.0
        Downloader.DOWNLOAD_TIMEOUT = 1.0
        self.downloader = Downloader(Path(self.tmpdir, ".downloads.db"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_failing_download(self):
        # fails with wrong url
        with self.assertRaises(DownloadFailed) as dfe:
            self.downloader.download(
                url="https://data.iac.ethz.ch/climada/9ff79b46-f912-41f8-b391-3f315689d246/"
                "OSM_features_48_8.cpg",  # instead of ...47_8...
                target_dir=self.tmpdir,
            )
        self.assertFalse(Path(self.tmpdir, "OSM_features_48_8.cpg").exists())

        with self.assertRaises(DownloadFailed) as dfe:
            self.downloader.download(
                url="https://data.iac.ethz.ch/climada/9ff79b46-f912-41f8-b391-3f315689d246/"
                "OSM_features_47_8.cpg",
                target_dir=self.tmpdir,
                size=77,
            )
        self.assertFalse(Path(self.tmpdir, "OSM_features_47_8.cpg").exists())

    def test_passing_download(self):
        before_download = time.time()
        dlf = self.downloader.download(
            url="https://data.iac.ethz.ch/climada/9ff79b46-f912-41f8-b391-3f315689d246/"
            "OSM_features_47_8.cpg",
            target_dir=self.tmpdir,
        )
        after_download = time.time()
        # the file is there
        self.assertTrue(Path(self.tmpdir, "OSM_features_47_8.cpg").is_file())
        # and is the same as the one returned
        self.assertTrue(dlf.is_file())
        self.assertEqual(dlf.name, "OSM_features_47_8.cpg")
        self.assertEqual(dlf.parent, Path(self.tmpdir))

        # check times
        timestamp = dlf.stat().st_mtime
        self.assertGreaterEqual(timestamp, before_download)
        self.assertLessEqual(timestamp, after_download)

        # download again
        dlf = self.downloader.download(
            url="https://data.iac.ethz.ch/climada/9ff79b46-f912-41f8-b391-3f315689d246/"
            "OSM_features_47_8.cpg",
            target_dir=self.tmpdir,
        )
        # there was nothing downloaded, file is the same as before
        self.assertEqual(timestamp, dlf.stat().st_mtime)

        with self.assertRaises(DownloadFailed) as rte:
            self.downloader.download(
                url="https://data.iac.ethz.ch/climada/9ff79b46-f912-41f8-b391-3f315689d246/"
                "OSM_features_48_8.cpg",
                target_dir=self.tmpdir,
                file_name="OSM_features_47_8.cpg",
            )
        self.assertIn(
            ") has been downloaded from another url before", str(rte.exception)
        )


class TestChecksum(unittest.TestCase):
    """Test file_checksum function"""

    def test_hashsum(self):
        self.assertEqual(
            "md5:7451535a7ec33ac056cd30b1a664fdcb",
            file_checksum(Path(__file__).parent / "__init__.py", "md5"),
        )
        self.assertEqual(
            "sha1:967e1de9d4f866f36c209dbd2aa52504a36553f7",
            file_checksum(Path(__file__).parent / "__init__.py", "sha1"),
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
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDownloader)
    TESTS = TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecksum))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestToStrList))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGetFileNames))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDownloadUrl))
    unittest.TextTestRunner(verbosity=2).run(TESTS)

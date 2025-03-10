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

Test save module.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from climada import CONFIG
from climada.util.api_client import Client, DataTypeShortInfo, Download

DATA_DIR = CONFIG.test_data.dir()


class TestClient(unittest.TestCase):
    """Test data api client methods."""

    def test_data_type(self):
        """"""
        lpdt = Client().get_data_type_info("tropical_cyclone")
        self.assertEqual(lpdt.data_type, "tropical_cyclone")
        self.assertEqual(lpdt.data_type_group, "hazard")
        self.assertTrue(
            "res_arcsec" in [p["property"] for p in lpdt.properties if p["mandatory"]]
        )
        self.assertTrue(
            "ref_year" in [p["property"] for p in lpdt.properties if not p["mandatory"]]
        )

    def test_data_types(self):
        """"""
        exdts = Client().list_data_type_infos("exposures")
        self.assertTrue("litpop" in [exdt.data_type for exdt in exdts])

    def test_datasets(self):
        """"""
        datasets = Client().list_dataset_infos(
            status=None, name="FAOSTAT_data_producer_prices"
        )
        self.assertEqual(len(datasets), 1)

    def test_dataset(self):
        """"""
        client = Client()

        dataset = client.get_dataset_info(
            name="FAOSTAT_data_producer_prices", status="test_dataset"
        )
        self.assertEqual(dataset.version, "v1")
        self.assertEqual(len(dataset.files), 1)
        self.assertEqual(dataset.files[0].file_size, 26481)
        self.assertEqual(
            dataset.data_type, DataTypeShortInfo("crop_production", "exposures")
        )

        dataset2 = client.get_dataset_info_by_uuid(dataset.uuid)
        self.assertEqual(dataset, dataset2)

    def test_search_for_property_not_set(self):
        """"""
        client = Client()

        nocountry = client.list_dataset_infos(
            data_type="earthquake", properties={"country_name": None}
        )[0]
        self.assertNotIn("country_name", nocountry.properties)
        self.assertIn("spatial_coverage", nocountry.properties)

    def test_dataset_offline(self):
        """"""
        client = Client()
        client.online = False

        with self.assertLogs("climada.util.api_client", level="WARNING") as cm:
            dataset = client.get_dataset_info(
                name="FAOSTAT_data_producer_prices", status="test_dataset"
            )
        self.assertIn(
            "there is no internet connection but the client has stored ", cm.output[0]
        )

        self.assertEqual(dataset.version, "v1")
        self.assertEqual(len(dataset.files), 1)
        self.assertEqual(dataset.files[0].file_size, 26481)
        self.assertEqual(
            dataset.data_type, DataTypeShortInfo("crop_production", "exposures")
        )

        with self.assertRaises(AssertionError) as ar:
            with self.assertLogs("climada.util.api_client", level="WARNING") as cm:
                dataset2 = Client().get_dataset_info_by_uuid(dataset.uuid)
        self.assertIn("no logs of level WARNING or higher triggered", str(ar.exception))
        self.assertEqual(dataset, dataset2)

        with self.assertLogs("climada.util.api_client", level="WARNING") as cm:
            dataset2 = client.get_dataset_info_by_uuid(dataset.uuid)
        self.assertIn(
            "there is no internet connection but the client has stored ", cm.output[0]
        )
        self.assertEqual(dataset, dataset2)

    def test_download_file(self):
        """"""
        client = Client()
        client.MAX_WAITING_PERIOD = 0.1
        dataset = client.get_dataset_info(
            name="FAOSTAT_data_producer_prices", status="test_dataset"
        )

        # test failure
        def fail(x, y):
            raise Download.Failed("on purpose")

        self.assertRaises(
            Download.Failed,
            client._download_file,
            DATA_DIR,
            dataset.files[0],
            check=fail,
        )
        self.assertFalse(DATA_DIR.joinpath(dataset.files[0].file_name).is_file())

        # test success
        download = client._download_file(DATA_DIR, dataset.files[0])
        self.assertEqual(download, DATA_DIR / dataset.files[0].file_name)
        self.assertTrue(download.is_file())
        self.assertEqual(download.stat().st_size, dataset.files[0].file_size)
        download.unlink()
        self.assertFalse(download.is_file())

    def test_download_dataset(self):
        """"""
        client = Client()
        client.MAX_WAITING_PERIOD = 0.1

        dataset = client.get_dataset_info(
            name="test_write_raster", status="test_dataset"
        )
        download_dir, downloads = client.download_dataset(dataset, target_dir=DATA_DIR)
        self.assertEqual(download_dir.name, dataset.version)
        self.assertEqual(download_dir.parent.name, dataset.name)
        self.assertEqual(download_dir.parent.parent.name, dataset.data_type.data_type)
        self.assertEqual(len(downloads), 2)
        for download, dsfile in zip(downloads, dataset.files):
            self.assertTrue(download.is_file())
            self.assertEqual(download.stat().st_size, dsfile.file_size)
            self.assertEqual(download.name, dsfile.file_name)
            self.assertEqual(download.parent, download_dir)
            download.unlink()
        rm_empty_dir(download.parent.parent.parent)

    def test_get_exposures(self):
        client = Client()
        exposures = client.get_exposures(
            exposures_type="litpop",
            properties={
                "country_iso3alpha": "AUT",
                "fin_mode": "pop",
                "exponents": "(0,1)",
            },
            version="v1",
            dump_dir=DATA_DIR,
        )
        self.assertEqual(len(exposures.gdf), 5782)
        self.assertEqual(np.unique(exposures.region_id), 40)
        self.assertEqual(
            exposures.description,
            "LitPop Exposure for ['AUT'] at 150 as, year: 2018, financial mode: pop, exp: [0, 1], admin1_calc: False",
        )

    def test_get_exposures_fails(self):
        client = Client()
        with self.assertRaises(ValueError) as cm:
            client.get_exposures(
                exposures_type="river_flood",
                properties={
                    "country_iso3alpha": "AUT",
                    "fin_mode": "pop",
                    "exponents": "(0,1)",
                },
                dump_dir=DATA_DIR,
            )
        self.assertIn(
            "Valid exposures types are a subset of CLIMADA exposures types. Currently",
            str(cm.exception),
        )

        with self.assertRaises(Client.AmbiguousResult) as cm:
            client.get_exposures(
                exposures_type="litpop",
                properties={"country_iso3alpha": "AUT"},
                dump_dir=DATA_DIR,
            )
        self.assertIn(
            "there are 3 datasets meeting the requirements", str(cm.exception)
        )

    def test_get_hazard(self):
        client = Client()
        hazard = client.get_hazard(
            hazard_type="river_flood",
            properties={
                "country_name": "Austria",
                "year_range": "2010_2030",
                "climate_scenario": "rcp26",
            },
            version="v1",
            dump_dir=DATA_DIR,
        )
        self.assertEqual(np.shape(hazard.intensity), (480, 5784))
        self.assertEqual(np.unique(hazard.centroids.region_id), 40)
        self.assertEqual(np.unique(hazard.date).size, 20)
        self.assertEqual(hazard.haz_type, "RF")

    def test_get_hazard_fails(self):
        client = Client()
        with self.assertRaises(ValueError) as cm:
            client.get_hazard(
                hazard_type="litpop",
                properties={
                    "country_name": "Austria",
                    "year_range": "2010_2030",
                    "climate_scenario": "rcp26",
                },
                dump_dir=DATA_DIR,
            )
        self.assertIn(
            "Valid hazard types are a subset of CLIMADA hazard types. Currently",
            str(cm.exception),
        )

        with self.assertRaises(Client.AmbiguousResult) as cm:
            client.get_hazard(
                hazard_type="river_flood",
                properties={
                    "country_name": ["Switzerland", "Austria"],
                    "year_range": "2010_2030",
                    "climate_scenario": ["rcp26", "rcp85"],
                },
                dump_dir=DATA_DIR,
            )
        self.assertIn(
            "there are 4 datasets meeting the requirements:", str(cm.exception)
        )

    def test_get_litpop(self):
        client = Client()
        litpop = client.get_litpop(country="LUX", version="v1", dump_dir=DATA_DIR)
        self.assertEqual(len(litpop.gdf), 188)
        self.assertEqual(np.unique(litpop.region_id), 442)
        self.assertEqual(
            litpop.description,
            "LitPop Exposure for ['LUX'] at 150 as, year: 2018, financial mode: pc, exp: [1, 1], admin1_calc: False",
        )

    def test_get_litpop_fail(self):
        client = Client()
        with self.assertRaises(ValueError) as cm:
            client.get_litpop(["AUT", "CHE"])
        self.assertIn(
            " can only query single countries. Download the data for multiple countries individually and concatenate ",
            str(cm.exception),
        )

    def test_get_centroids_plot(self):
        client = Client()
        client.get_centroids(country="COM").plot()

    def test_get_dataset_file(self):
        client = Client()
        with tempfile.TemporaryDirectory() as temp_dir:
            single_file = client.get_dataset_file(
                name="test_imp_mat",
                status="test_dataset",  # get_dataset_info arguments
                target_dir=Path(temp_dir),
                organize_path=False,  # download_dataset arguments
            )
            self.assertTrue(single_file.is_file())
            self.assertEqual(list(Path(temp_dir).iterdir()), [single_file])

    def test_multi_filter(self):
        client = Client()
        testds = client.list_dataset_infos(data_type="storm_europe")

        # assert no systemic loss in filtering
        still = client._filter_datasets(testds, dict())
        for o, r in zip(testds, still):
            self.assertEqual(o, r)

        # assert filter is effective
        p = "country_name"
        a, b = "Germany", "Netherlands"
        less = client._filter_datasets(testds, {p: [a, b]})
        self.assertLess(len(less), len(testds))
        only = client._filter_datasets(testds, {p: [a]})
        self.assertLess(len(only), len(less))
        self.assertLess(0, len(only))

    def test_multiplicity_split(self):
        properties = {"country_name": ["x", "y", "z"], "b": "1"}
        # assert split matches expectations
        straight, multi = Client._divide_straight_from_multi(properties)
        self.assertEqual(straight, {"b": "1"})
        self.assertEqual(multi, {"country_name": ["x", "y", "z"]})

    def test_purge_cache(self):
        client = Client()

        active_ds = client.get_dataset_info(
            data_type="litpop", name="LitPop_150arcsec_ABW", version="v3"
        )
        outdated_ds = client.get_dataset_info(
            data_type="litpop", name="LitPop_150arcsec_ABW", version="v1"
        )
        test_ds = client.get_dataset_info(
            data_type="storm_europe",
            name="test_storm_europe_icon_2021012800",
            version="v1",
            status="test_dataset",
        )
        expired_ds = client.get_dataset_info(
            data_type="tropical_cyclone",
            name="rename_files2",
            version="v1",
            status="expired",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            for ds in [active_ds, outdated_ds, test_ds, expired_ds]:
                client.download_dataset(dataset=ds, target_dir=Path(temp_dir))
            self.assertEqual(  # outdated dataset present
                1,
                len(
                    list(
                        Path(temp_dir)
                        .joinpath("exposures/litpop/LitPop_150arcsec_ABW/v1")
                        .iterdir()
                    )
                ),
            )
            self.assertEqual(  # expired data set present
                1,
                len(
                    list(
                        Path(temp_dir)
                        .joinpath("hazard/tropical_cyclone/rename_files2/v1")
                        .iterdir()
                    )
                ),
            )

            client.purge_cache(target_dir=temp_dir)
            self.assertFalse(  # outdated data set removed
                Path(temp_dir)
                .joinpath("exposures/litpop/LitPop_150arcsec_ABW/v1")
                .is_dir()
            )
            self.assertFalse(  # expired data set removed
                Path(temp_dir)
                .joinpath("hazard/tropical_cyclone/rename_files2/v1")
                .is_dir()
            )
            self.assertEqual(  # test files are still there
                3,
                len(
                    list(
                        Path(temp_dir)
                        .joinpath(
                            "hazard/storm_europe/test_storm_europe_icon_2021012800/v1"
                        )
                        .iterdir()
                    )
                ),
            )

            client.purge_cache(target_dir=temp_dir, keep_testfiles=False)
            self.assertTrue(  # uptodate active dataset file still there
                Path(temp_dir)
                .joinpath(
                    "exposures/litpop/LitPop_150arcsec_ABW/v3/LitPop_150arcsec_ABW.hdf5"
                )
                .exists()
            )
            self.assertFalse(  # test data removed, empty directories removed
                Path(temp_dir).joinpath("hazard/").exists()
            )


def rm_empty_dir(folder):
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            rm_empty_dir(subfolder)
    try:
        folder.rmdir()
    except:
        pass


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestClient)
    unittest.TextTestRunner(verbosity=2).run(TESTS)

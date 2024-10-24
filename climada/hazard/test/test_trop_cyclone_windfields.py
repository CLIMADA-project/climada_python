import unittest

import numpy as np
import xarray as xr

from climada.hazard import TCTracks
from climada.hazard.test.test_trop_cyclone import TEST_TRACK, TEST_TRACK_SHORT
from climada.hazard.trop_cyclone.trop_cyclone_windfields import (
    H_TO_S,
    KM_TO_M,
    MBAR_TO_PA,
    _B_holland_1980,
    _bs_holland_2008,
    _stat_er_2011,
    _stat_holland_1980,
    _stat_holland_2010,
    _v_max_s_holland_2008,
    _vtrans,
    _x_holland_2010,
    get_close_centroids,
    tctrack_to_si,
)
from climada.util import ureg


class TestWindfieldHelpers(unittest.TestCase):
    """Test helper functions of TC wind field model"""

    def test_get_close_centroids_pass(self):
        """Test get_close_centroids function."""
        si_track = xr.Dataset(
            {
                "lat": ("time", np.array([0, -0.5, 0])),
                "lon": ("time", np.array([0.9, 2, 3.2])),
            },
            attrs={"mid_lon": 0.0},
        )
        centroids = np.array(
            [
                [0, -0.2],
                [0, 0.9],
                [-1.1, 1.2],
                [1, 2.1],
                [0, 4.3],
                [0.6, 3.8],
                [0.9, 4.1],
            ]
        )
        centroids_close, mask_close, mask_close_alongtrack = get_close_centroids(
            si_track, centroids, 112.0
        )
        self.assertEqual(centroids_close.shape[0], mask_close.sum())
        self.assertEqual(mask_close_alongtrack.shape[0], si_track.sizes["time"])
        self.assertEqual(mask_close_alongtrack.shape[1], centroids_close.shape[0])
        np.testing.assert_equal(mask_close_alongtrack.any(axis=0), True)
        np.testing.assert_equal(
            mask_close, np.array([False, True, True, False, False, True, False])
        )
        np.testing.assert_equal(
            mask_close_alongtrack,
            np.array(
                [
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                ]
            ),
        )
        np.testing.assert_equal(centroids_close, centroids[mask_close])

        # example where antimeridian is crossed
        si_track = xr.Dataset(
            {
                "lat": ("time", np.linspace(-10, 10, 11)),
                "lon": ("time", np.linspace(170, 200, 11)),
            },
            attrs={"mid_lon": 180.0},
        )
        centroids = np.array([[-11, 169], [-7, 176], [4, -170], [10, 170], [-10, -160]])
        centroids_close, mask_close, mask_close_alongtrack = get_close_centroids(
            si_track, centroids, 600.0
        )
        self.assertEqual(centroids_close.shape[0], mask_close.sum())
        self.assertEqual(mask_close_alongtrack.shape[0], si_track.sizes["time"])
        self.assertEqual(mask_close_alongtrack.shape[1], centroids_close.shape[0])
        np.testing.assert_equal(mask_close_alongtrack.any(axis=0), True)
        np.testing.assert_equal(mask_close, np.array([True, True, True, False, False]))
        np.testing.assert_equal(
            centroids_close,
            np.array(
                [
                    # the longitudinal coordinate of the third centroid is normalized
                    [-11, 169],
                    [-7, 176],
                    [4, 190],
                ]
            ),
        )

    def test_B_holland_1980_pass(self):
        """Test _B_holland_1980 function."""
        si_track = xr.Dataset(
            {
                "pdelta": ("time", MBAR_TO_PA * np.array([15, 30])),
                "vgrad": ("time", [35, 40]),
                "rho_air": ("time", [1.15, 1.15]),
            }
        )
        _B_holland_1980(si_track)
        np.testing.assert_array_almost_equal(si_track["hol_b"], [2.5, 1.667213])

        si_track = xr.Dataset(
            {
                "pdelta": ("time", MBAR_TO_PA * np.array([4.74, 15, 30, 40])),
                "vmax": ("time", [np.nan, 22.5, 25.4, 42.5]),
                "rho_air": ("time", [1.2, 1.2, 1.2, 1.2]),
            }
        )
        _B_holland_1980(si_track, gradient_to_surface_winds=0.9)
        np.testing.assert_allclose(
            si_track["hol_b"], [np.nan, 1.101, 0.810, 1.473], atol=1e-3
        )

    def test_bs_holland_2008_pass(self):
        """Test _bs_holland_2008 function. Compare to MATLAB reference."""
        si_track = xr.Dataset(
            {
                "tstep": ("time", H_TO_S * np.array([1.0, 1.0, 1.0])),
                "lat": (
                    "time",
                    [12.299999504631234, 12.299999504631343, 12.299999279463769],
                ),
                "pdelta": ("time", MBAR_TO_PA * np.array([4.74, 4.73, 4.73])),
                "cen": (
                    "time",
                    MBAR_TO_PA * np.array([1005.2585, 1005.2633, 1005.2682]),
                ),
                "vtrans_norm": ("time", [np.nan, 5.241999541820597, 5.123882725120426]),
            }
        )
        _bs_holland_2008(si_track)
        np.testing.assert_allclose(si_track["hol_b"], [np.nan, 1.27, 1.27], atol=1e-2)

    def test_v_max_s_holland_2008_pass(self):
        """Test _v_max_s_holland_2008 function."""
        # Numbers analogous to test_B_holland_1980_pass
        si_track = xr.Dataset(
            {
                "pdelta": ("time", MBAR_TO_PA * np.array([15, 30])),
                "hol_b": ("time", [2.5, 1.67]),
                "rho_air": ("time", [1.15, 1.15]),
            }
        )
        _v_max_s_holland_2008(si_track)
        np.testing.assert_array_almost_equal(si_track["vmax"], [34.635341, 40.033421])

    def test_holland_2010_pass(self):
        """Test Holland et al. 2010 wind field model."""
        # The parameter "x" is designed to be exactly 0.5 inside the radius of max wind (RMW) and
        # to increase or decrease linearly outside of it in radial direction.
        #
        # An increase (decrease) of "x" outside of the RMW is for cases where the max wind is very
        # high (low), but the RMW is still comparably large (small). This means, wind speeds need
        # to decay very sharply (only moderately) outside of the RMW to reach the low prescribed
        # peripheral wind speeds.
        #
        # The "hol_b" parameter tunes the meaning of a "comparably" large or small RMW.
        si_track = xr.Dataset(
            {
                # four test cases:
                # - low vmax, moderate RMW: x decreases moderately
                # - large hol_b: x decreases sharply
                # - very low vmax: x decreases so much, it needs to be clipped at 0
                # - large vmax, large RMW: x increases
                "rad": ("time", KM_TO_M * np.array([75, 75, 75, 90])),
                "vmax": ("time", [35.0, 35.0, 16.0, 90.0]),
                "hol_b": ("time", [1.75, 2.5, 1.9, 1.6]),
            }
        )
        d_centr = KM_TO_M * np.array(
            [
                # first column is for locations within the storm eye
                # second column is for locations at or close to the radius of max wind
                # third column is for locations outside the storm eye
                # fourth column is for locations exactly at the peripheral radius
                # fifth column is for locations outside the peripheral radius
                [0.0, 75, 220, 300, 490],
                [30, 74, 170, 300, 501],
                [21, 76, 230, 300, 431],
                [32, 91, 270, 300, 452],
            ],
            dtype=float,
        )
        close_centr = np.array(
            [
                # note that we set one of these to "False" for testing
                [True, True, True, True, True],
                [True, True, True, True, False],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ],
            dtype=bool,
        )
        hol_x = _x_holland_2010(si_track, d_centr, close_centr)
        np.testing.assert_array_almost_equal(
            hol_x,
            [
                [0.5, 0.500000, 0.485077, 0.476844, 0.457291],
                [0.5, 0.500000, 0.410997, 0.400000, 0.000000],
                [0.5, 0.497620, 0.400000, 0.400000, 0.400000],
                [0.5, 0.505022, 1.403952, 1.554611, 2.317948],
            ],
        )

        v_ang_norm = _stat_holland_2010(si_track, d_centr, close_centr, hol_x)
        np.testing.assert_allclose(
            v_ang_norm,
            [
                # first column: converge to 0 when approaching storm eye
                # second column: vmax at RMW
                # fourth column: peripheral speed (17 m/s) at peripheral radius (unless x is clipped!)
                [0.000000, 35.000000, 21.181497, 17.000000, 12.1034610],
                [1.296480, 34.990037, 21.593755, 12.891313, 0.0000000],
                [0.321952, 15.997500, 9.712006, 8.087240, 6.2289690],
                [24.823469, 89.992938, 24.381965, 17.000000, 1.9292020],
            ],
            atol=1e-6,
        )

    def test_stat_holland_1980(self):
        """Test _stat_holland_1980 function. Compare to MATLAB reference."""
        d_centr = KM_TO_M * np.array(
            [
                [
                    299.4501244109841,
                    291.0737897183741,
                    292.5441003235722,
                    40.665454622610511,
                ],
                [293.6067129546862, 1000.0, 298.2652319413182, 70.0],
            ]
        )
        si_track = xr.Dataset(
            {
                "rad": (
                    "time",
                    KM_TO_M * np.array([40.665454622610511, 75.547902916671745]),
                ),
                "hol_b": ("time", [1.486076257880692, 1.265551666104679]),
                "pdelta": ("time", MBAR_TO_PA * np.array([39.12, 4.73])),
                "lat": ("time", [-14.089110370469488, 12.299999279463769]),
                "cp": ("time", [3.54921922e-05, 3.10598285e-05]),
                "rho_air": ("time", [1.15, 1.15]),
            }
        )
        mask = np.array(
            [[True, True, True, True], [True, False, True, True]], dtype=bool
        )
        v_ang_norm = _stat_holland_1980(si_track, d_centr, mask)
        np.testing.assert_allclose(
            v_ang_norm,
            [[11.28, 11.68, 11.61, 42.41], [5.38, 0, 5.28, 12.76]],
            atol=1e-2,
        )

        # without Coriolis force, values are higher, esp. far away from the center:
        v_ang_norm = _stat_holland_1980(si_track, d_centr, mask, cyclostrophic=True)
        np.testing.assert_allclose(
            v_ang_norm,
            [[15.72, 16.04, 15.98, 43.13], [8.84, 0, 8.76, 13.81]],
            atol=1e-2,
        )

        d_centr = np.array([[], []])
        mask = np.ones_like(d_centr, dtype=bool)
        v_ang_norm = _stat_holland_1980(si_track, d_centr, mask)
        np.testing.assert_array_equal(v_ang_norm, np.array([[], []]))

    def test_er_2011_pass(self):
        """Test Emanuel and Rotunno 2011 wind field model."""
        # test at centroids within and outside of radius of max wind
        d_centr = KM_TO_M * np.array(
            [[35, 70, 75, 220], [30, 150, 1000, 300]], dtype=float
        )
        si_track = xr.Dataset(
            {
                "rad": ("time", KM_TO_M * np.array([75.0, 40.0])),
                "vmax": ("time", [35.0, 40.0]),
                "lat": ("time", [20.0, 27.0]),
                "cp": ("time", [4.98665369e-05, 6.61918149e-05]),
            }
        )
        mask = np.array(
            [[True, True, True, True], [True, False, True, True]], dtype=bool
        )
        v_ang_norm = _stat_er_2011(si_track, d_centr, mask)
        np.testing.assert_array_almost_equal(
            v_ang_norm,
            [
                [28.258025, 36.782418, 36.869995, 22.521237],
                [39.670883, 0, 3.300626, 10.827206],
            ],
        )

    def test_vtrans_pass(self):
        """Test _vtrans function. Compare to MATLAB reference."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        track_ds = tc_track.data[0]

        si_track = tctrack_to_si(track_ds)
        _vtrans(si_track)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude

        self.assertEqual(si_track["vtrans_norm"].size, track_ds["time"].size)
        self.assertEqual(si_track["vtrans_norm"].values[0], 0)
        self.assertAlmostEqual(si_track["vtrans_norm"].values[1] * to_kn, 10.191466246)

    def testtctrack_to_si(self):
        """Test tctrack_to_si should create the same vmax output independent of the input unit"""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT).data[0]

        tc_track_kmph = tc_track.copy(deep=True)
        tc_track_kmph["max_sustained_wind"] *= (
            (1.0 * ureg.knot).to(ureg.km / ureg.hour).magnitude
        )
        tc_track_kmph.attrs["max_sustained_wind_unit"] = "km/h"

        si_track = tctrack_to_si(tc_track)
        si_track_from_kmph = tctrack_to_si(tc_track_kmph)

        np.testing.assert_array_almost_equal(
            si_track["vmax"], si_track_from_kmph["vmax"]
        )

        tc_track.attrs["max_sustained_wind_unit"] = "elbows/fortnight"
        with self.assertRaises(ValueError):
            tctrack_to_si(tc_track)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestWindfieldHelpers)
    unittest.TextTestRunner(verbosity=2).run(TESTS)

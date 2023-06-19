import unittest

from climada.util.multi_risk import *
from climada.entity import INDICATOR_IMPF, Exposures, ImpactFunc
from climada.hazard import Hazard, Centroids
from climada.util import DEF_CRS
import numpy as np
import geopandas as gpd
import sparse
import scipy
from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHaz
from climada.entity import ImpactFuncSet


def dummy_exp1(id=1):
    """Followng values are defined for each exposure"""
    data = {}
    data['latitude'] = np.array([1, 2, 2])
    data['longitude'] = np.array([2, 3, 2])
    data['value'] = np.array([1, 2, 3])
    data['deductible'] = np.array([1, 2, 3])
    data[INDICATOR_IMPF] = np.array([id, id, id])
    data['category_id'] = np.array([1, 2, 3])
    data['region_id'] = np.array([1, 2, 3])
    expo = Exposures(gpd.GeoDataFrame(data=data))
    return expo


def dummy_exp2(id=1):
    """Followng values are defined for each exposure"""
    data = {}
    data['latitude'] = np.array([1, 8])
    data['longitude'] = np.array([2, 8])
    data['value'] = np.array([1, 2])
    data['deductible'] = np.array([1, 2])
    data[INDICATOR_IMPF] = np.array([id, id])
    data['category_id'] = np.array([1, 2])
    data['region_id'] = np.array([1, 2])
    expo = Exposures(gpd.GeoDataFrame(data=data))
    return expo

def dummy_impf1(id, haz_type):
    inten = (0, 100, 5)
    imp_fun = ImpactFunc.from_sigmoid_impf(
        inten, L=1.0, k=2., x0=0.5, haz_type=haz_type, impf_id=id)
    impf_set = ImpactFuncSet(impact_funcs=[imp_fun])
    return impf_set

def dummy_haz1(haz_type='H1'):
    fraction = None
    years = np.arange(2014, 2016)
    dates = np.array([datetime.date(year, 1, 1).toordinal() for year in years])
    event_ids = np.arange(1, len(dates)+1)
    centroids = Centroids.from_lat_lon(
        np.array([1, 2, 2]), np.array([2, 3, 2]))
    intensity = scipy.sparse.csr_matrix(np.array(
            [[1, 0, 1, 2],
             [2, 2, 1,  2]]
            ))
    intensity.data = intensity.data*100
    frequency = np.full(len(event_ids), 1 / len(years))

    hazard = Hazard(
        haz_type,
        intensity=intensity,
        fraction=fraction,
        centroids=centroids,
        event_id=event_ids,
        event_name=event_ids,
        date=dates,
        orig=np.array([True, False, False, True]),
        frequency=frequency,
        frequency_unit='1/year',
        units='m/s',
        file_name="file1.mat",
        description="Description 1",
    )
    return hazard

def dummy_haz2():
    years = np.arange(2014, 2016)  # Update the range as needed
    months = np.arange(1, 13)
    event_ids = np.arange(1, 25)  # Two years with one event per month
    dates = np.array([datetime.date(year, month, 1).toordinal() for year in years for month in months])
    fraction = None
    centroids = centroids = Centroids.from_lat_lon(
        np.array([1, 2, 2]), np.array([2, 3, 2]))
    intensity = scipy.sparse.random(len(event_ids), len(centroids.coord), density=0.3, format='csr')
    intensity.data = intensity.data * 100
    frequency = np.full(len(event_ids), 1 / len(years))
    hazard = Hazard(
        "H2",
        intensity=intensity,
        fraction=fraction,
        centroids=centroids,
        event_id=event_ids,
        event_name=event_ids,
        date=dates,
        orig=np.array([True, False, False, True]),
        frequency=frequency,
        frequency_unit='1/year',
        units='m/s',
        file_name="file1.mat",
        description="Description 1",
    )
    return hazard

def dummy_haz3():
    years = np.arange(2014, 2016)  # Update the range as needed
    months = np.arange(1, 13)
    days = np.arange(1, 32)
    dates = np.array(
        [datetime.date(year, month, day).toordinal() for year in years for month in months for day in days])
    event_ids = np.arange(1, len(dates))  # Two years with one event per day
    centroids = Centroids.from_lat_lon(
        np.array([1, 2, 8]), np.array([2, 3, 8]))

    intensity = sparse.random((len(event_ids), len(centroids.coord)), density=0.3, format='csr')
    intensity.data = intensity.data * 100
    frequency = np.full(len(event_ids), 1 / len(years))  # One event per day
    hazard = Hazard(
        "H3",
        intensity=intensity,
        fraction=None,
        centroids=centroids,
        event_id=event_ids,
        event_name=event_ids,
        date=dates,
        orig=np.array([True, False, False, True]),
        frequency=frequency,
        frequency_unit='1/year',
        units='m/s',
        file_name="file1.mat",
        description="Description 1",
    )
    return hazard


def dummy_imp1():
    """Return an impact object for testing"""
    years = np.arange(2014, 2016)
    event_ids = np.arange(2, 4)
    event_name = event_ids.copy()
    coord_exp = np.array([[x, y] for x in np.linspace(1, 2, 2) for y in np.linspace(2, 3, 2)])
    dates = np.array([datetime.date(year, 1, 1).toordinal() for year in years])
    imp_mat = scipy.sparse.csr_matrix(np.array(
            [[1, 0, 1, 2],
             [2, 2, 1,  2]]
            ))
    frequency = np.full(len(event_ids), 1 / len(years))
    impact = Impact(
        event_id=event_ids,
        event_name=event_name,
        at_event=np.zeros(len(event_ids)),
        eai_exp=np.zeros(len(coord_exp)),
        date=dates,
        coord_exp=coord_exp,
        crs=DEF_CRS,
        imp_mat=imp_mat,
        frequency=frequency,
        tot_value=5,
        unit="USD",
        frequency_unit="1/year",
        tag={
            "exp": Tag("file_exp.p", "descr exp"),
            "haz": TagHaz("TC", "file_haz.p", "descr haz"),
            "impf_set": Tag(),
        },
    )
    impact.at_event, impact.eai_exp, impact.aai_agg = ImpactCalc.risk_metrics(impact.imp_mat, impact.frequency)
    return impact


def dummy_imp2():
    """Return an impact object for testing"""
    years = np.arange(2014, 2016)  # Update the range as needed
    months = np.arange(1, 3)
    days = np.arange(1, 15, 12)  # Adjust the number of days as needed
    dates = np.array([datetime.date(year, month, day).toordinal()
                      for year in years for month in months for day in days])
    event_ids = np.arange(1, len(dates)+1)  # Two years with one event per month
    event_names = event_ids.copy()

    coord_exp = np.array([[x, y] for x in np.linspace(1, 8, 2) for y in np.linspace(2, 8, 1)])
    crs = DEF_CRS  # Update with your CRS

    imp_mat = scipy.sparse.csr_matrix(np.array(
            [[1, 0],
             [2, 2],
             [3, 6],
             [3, 5],
             [1, 0],
             [1, 9],
             [3, 3],
             [0, 0]]
            ))

    frequency = np.full(len(event_ids), 1 / len(years))

    impact = Impact(
        event_id=event_ids,
        event_name=event_names,
        date=dates,
        coord_exp=coord_exp,
        crs=crs,
        imp_mat=imp_mat,
        frequency=frequency,
        tot_value=7,
        unit="USD",
        frequency_unit="1/month",
        at_event=np.zeros(len(event_ids)),
        eai_exp=np.zeros(len(coord_exp)),
        tag={
            "exp": Tag("file_exp.p", "descr exp"),
            "haz": TagHaz("TC", "file_haz.p", "descr haz"),
            "impf_set": Tag(),
        },
    )
    impact.at_event, impact.eai_exp, impact.aai_agg = ImpactCalc.risk_metrics(impact.imp_mat, impact.frequency)
    return impact


def dummy_imp3():
    """Return an impact object for testing"""
    years = np.arange(2014, 2016)
    event_ids = np.arange(2, 4)
    event_name = event_ids.copy()
    coord_exp = np.array([[x, y] for x in np.linspace(1, 2, 2) for y in np.linspace(2, 3, 2)])
    dates = np.array([datetime.date(year, 1, 1).toordinal() for year in years])
    imp_mat = scipy.sparse.csr_matrix(np.array(
        [[0, 0, 1, 0],
         [0, 0, 0, 1]]
    ))
    frequency = np.full(len(event_ids), 1 / len(years))
    impact = Impact(
        event_id=event_ids,
        event_name=event_name,
        at_event=np.zeros(len(event_ids)),
        eai_exp=np.zeros(len(coord_exp)),
        date=dates,
        coord_exp=coord_exp,
        crs=DEF_CRS,
        imp_mat=imp_mat,
        frequency=frequency,
        tot_value=5,
        unit="USD",
        frequency_unit="1/year",
        tag={
            "exp": Tag("file_exp.p", "descr exp"),
            "haz": TagHaz("TC", "file_haz.p", "descr haz"),
            "impf_set": Tag(),
        },
    )
    impact.at_event, impact.eai_exp, impact.aai_agg = ImpactCalc.risk_metrics(impact.imp_mat, impact.frequency)
    return impact

def dummy_imp4():
    """Return an impact object for testing"""
    years = np.arange(2014, 2016)
    event_ids = np.arange(2, 4)
    event_name = event_ids.copy()
    coord_exp = np.array([[x, y] for x in np.linspace(1, 2, 2) for y in np.linspace(2, 3, 2)])
    dates = np.array([datetime.date(year, 1, 1).toordinal() for year in years])
    imp_mat = scipy.sparse.csr_matrix(np.array(
        [[0, 1, 0, 0],
         [0, 0, 0, 1]]
    ))
    frequency = np.full(len(event_ids), 1 / len(years))
    impact = Impact(
        event_id=event_ids,
        event_name=event_name,
        at_event=np.zeros(len(event_ids)),
        eai_exp=np.zeros(len(coord_exp)),
        date=dates,
        coord_exp=coord_exp,
        crs=DEF_CRS,
        imp_mat=imp_mat,
        frequency=frequency,
        tot_value=5,
        unit="USD",
        frequency_unit="1/year",
        tag={
            "exp": Tag("file_exp.p", "descr exp"),
            "haz": TagHaz("TC", "file_haz.p", "descr haz"),
            "impf_set": Tag(),
        },
    )
    impact.at_event, impact.eai_exp, impact.aai_agg = ImpactCalc.risk_metrics(impact.imp_mat, impact.frequency)
    return impact

class TestMultiRisk(unittest.TestCase):
    def test_find_common_time_def(self):
        imp1 = dummy_imp1()
        imp2 = dummy_imp2()
        common_time_def = find_common_time_definition([imp1.date, imp2.date])

    def test_aggregate_events_by_date(self):
        imp = dummy_imp2()
        exp = dummy_exp2()
        imp_aggr_by_year = aggregate_impact_by_date(imp, by='year', exp=exp)
        comparison = np.all(imp_aggr_by_year.imp_mat.data <= np.array(exp.gdf.value[imp_aggr_by_year.imp_mat.nonzero()[1]]))
        self.assertTrue(comparison)
        imp_aggr_by_month = aggregate_impact_by_date(imp, by='month')
        self.assertAlmostEqual(imp_aggr_by_month.imp_mat.sum(), imp.imp_mat.sum())
        imp_aggr_by_weeks = aggregate_impact_by_date(imp, by='week')
        self.assertTrue(len(imp_aggr_by_weeks.date) == 8)
        self.assertTrue(np.all(imp_aggr_by_weeks.imp_mat.data == np.array([1, 2, 2, 6, 3, 5, 3, 1, 9, 1, 3, 3])))

    def test_fill_impact_gaps(self):
        imp1 = dummy_imp1()
        imp2 = dummy_imp2()
        filled_impacts = fill_impact_gaps({'imp1': imp1, 'imp2': imp2})

        # Get all unique dates and coordinates from the original impacts
        all_dates = sorted(set(date for imp in [imp1, imp2] for date in imp.date))
        all_coords = sorted(set(tuple(coord) for imp in [imp1, imp2] for coord in imp.coord_exp))

        # Verify that each filled impact has the correct shape
        for hazard in filled_impacts:
            self.assertEqual(filled_impacts[hazard].imp_mat.shape, (len(all_dates), len(all_coords)))

        # Verify that each filled impact has the correct date and coord_exp attributes
        for hazard, original_imp in zip(filled_impacts, [imp1, imp2]):
            self.assertEqual(filled_impacts[hazard].imp_mat.shape[0],
                             len(np.unique([date for imp in [imp1, imp2] for date in imp.date])))
            self.assertEqual(filled_impacts[hazard].imp_mat.shape[1],
                             len(np.unique([coord for imp in [imp1, imp2] for coord in imp.coord_exp],axis=0)))

            # Verify that each filled impact has the same aai_agg as the original impact
            self.assertAlmostEqual(filled_impacts[hazard].aai_agg, original_imp.aai_agg)

            # Verify that the sum of the imp_mat elements is still the same
            self.assertAlmostEqual(sum(filled_impacts[hazard].imp_mat.data), sum(original_imp.imp_mat.data))

    def test_combine_impacts(self):
        imp1 = dummy_imp1()
        imp_combined = combine_impacts([imp1, imp1])
        self.assertAlmostEqual(sum(imp_combined.imp_mat.data), sum(imp1.imp_mat.data)*2)
        imp2 = copy.deepcopy(imp1)
        imp2.date = imp2.date+1
        self.assertRaises(ValueError, combine_impacts, [imp1, imp2], by='date')
        self.assertRaises(NotImplementedError, combine_impacts, [imp1, imp2], by='e')

    def test_mask_single_hazard_impact(self):
        imp1 = dummy_imp1()
        imp2 = dummy_imp3()
        imp3 = dummy_imp4()
        compound_impact = mask_single_hazard_impact(imp1, [imp2, imp3])
        np.testing.assert_array_equal(compound_impact.imp_mat.todense(), [[0, 0, 0, 0], [0, 0, 0, 2]])
"""
Define MeasuresMat class.
"""

__all__ = ['MeasuresMat']

import numpy as np

import climada.util.hdf5_handler as hdf5
from climada.entity.measures.base import Measure, Measures
from climada.entity.tag import Tag

class MeasuresMat(Measures):
    """Measures class loaded from an excel file.

    Attributes
    ----------
        sup_field_name (str): name of the enclosing variable, if present
        sheet_name (str): name of excel sheet containing the data
        var (dict): name of the table columns for each of the attributes
    """

    def __init__(self, file_name=None, description=None):
        """Extend Measures __init__ method.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Examples
        --------
            >>> MeasuresMat()
            Initializes empty attributes.
            >>> MeasuresMat('filename')
            Loads data from the provided file.
            >>> MeasuresMat('filename', 'description of file')
            Loads data from the provided file and stores provided description.
        """
        self.sup_field_name = 'entity'
        self.field_name = 'measures'
        self.var = {'name' : 'name',
                    'color' : 'color',
                    'cost' : 'cost',
                    'haz_int_a' : 'hazard_intensity_impact_a',
                    'haz_int_b' : 'hazard_intensity_impact_b',
                    'haz_frq' : 'hazard_high_frequency_cutoff',
                    'haz_set' : 'hazard_event_set',
                    'mdd_a' : 'MDD_impact_a',
                    'mdd_b' : 'MDD_impact_b',
                    'paa_a' : 'PAA_impact_a',
                    'paa_b' : 'PAA_impact_b',
                    'risk_att' : 'risk_transfer_attachement',
                    'risk_cov' : 'risk_transfer_cover'
                   }
        # Initialize
        Measures.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """Override read Loader method."""
        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # Load mat data
        meas_all = hdf5.read(file_name)
        try:
            meas_all = meas_all[self.sup_field_name]
        except KeyError:
            pass
        meas_all = meas_all[self.field_name]

        # number of measures
        num_mes = len(meas_all[self.var['name']])

        # iterate over each measure
        for idx in range(0, num_mes):
            meas = Measure()

            meas.name = hdf5.get_str_from_ref(
                file_name, meas_all[self.var['name']][idx][0])

            color_str = hdf5.get_str_from_ref(
                file_name, meas_all[self.var['color']][idx][0])
            meas.color_rgb = np.fromstring(color_str, dtype=float, sep=' ')
            meas.cost = meas_all[self.var['cost']][idx][0]
            meas.hazard_freq_cutoff = meas_all[self.var['haz_frq']][idx][0]
            meas.hazard_event_set = hdf5.get_str_from_ref(
                file_name, meas_all[self.var['haz_set']][idx][0])
            meas.hazard_intensity = (meas_all[self.var['haz_int_a']][idx][0], \
                                     meas_all[self.var['haz_int_b']][0][idx])
            meas.mdd_impact = (meas_all[self.var['mdd_a']][idx][0],
                               meas_all[self.var['mdd_b']][idx][0])
            meas.paa_impact = (meas_all[self.var['paa_a']][idx][0],
                               meas_all[self.var['paa_b']][idx][0])
            meas.risk_transf_attach = meas_all[self.var['risk_att']][idx][0]
            meas.risk_transf_cover = meas_all[self.var['risk_cov']][idx][0]

            self.data.append(meas)

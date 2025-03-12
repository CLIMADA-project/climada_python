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

Define ImpactFunc class.
"""

__all__ = ["ImpactFunc"]

import logging
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

import climada.util.checker as u_check

LOGGER = logging.getLogger(__name__)


class ImpactFunc:
    """Contains the definition of one impact function.

    Attributes
    ----------
    haz_type : str
        hazard type acronym (e.g. 'TC')
    id : int or str
        id of the impact function. Exposures of the same type
        will refer to the same impact function id
    name : str
        name of the ImpactFunc
    intensity_unit : str
        unit of the intensity
    intensity : np.array
        intensity values
    mdd : np.array
        mean damage (impact) degree for each intensity (numbers
        in [0,1])
    paa : np.array
        percentage of affected assets (exposures) for each
        intensity (numbers in [0,1])
    """

    def __init__(
        self,
        haz_type: str = "",
        id: Union[str, int] = "",
        intensity: Optional[np.ndarray] = None,
        mdd: Optional[np.ndarray] = None,
        paa: Optional[np.ndarray] = None,
        intensity_unit: str = "",
        name: str = "",
    ):
        """Initialization.

        Parameters
        ----------
        haz_type : str, optional
            Hazard type acronym (e.g. 'TC').
        id : int or str, optional
            id of the impact function. Exposures of the same type
            will refer to the same impact function id.
        intensity : np.array, optional
            Intensity values. Defaults to empty array.
        mdd : np.array, optional
            Mean damage (impact) degree for each intensity (numbers
            in [0,1]). Defaults to empty array.
        paa : np.array, optional
            Percentage of affected assets (exposures) for each
            intensity (numbers in [0,1]). Defaults to empty array.
        intensity_unit : str, optional
            Unit of the intensity.
        name : str, optional
            Name of the ImpactFunc.
        """
        self.id = id
        self.name = name
        self.intensity_unit = intensity_unit
        self.haz_type = haz_type
        # Followng values defined for each intensity value
        self.intensity = intensity if intensity is not None else np.array([])
        self.mdd = mdd if mdd is not None else np.array([])
        self.paa = paa if paa is not None else np.array([])

    def calc_mdr(self, inten: Union[float, np.ndarray]) -> np.ndarray:
        """Interpolate impact function to a given intensity.

        Parameters
        ----------
        inten : float or np.array
            intensity, the x-coordinate of the
            interpolated values.

        Returns
        -------
        np.array
        """
        #        return np.interp(inten, self.intensity, self.mdd * self.paa)
        return np.interp(inten, self.intensity, self.paa) * np.interp(
            inten, self.intensity, self.mdd
        )

    def plot(self, axis=None, **kwargs):
        """Plot the impact functions MDD, MDR and PAA in one graph, where
        MDR = PAA * MDD.

        Parameters
        ----------
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs : optional
            arguments for plot matplotlib function, e.g. marker='x'

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        if axis is None:
            _, axis = plt.subplots(1, 1)

        title = "%s %s" % (self.haz_type, str(self.id))
        if self.name != str(self.id):
            title += ": %s" % self.name
        axis.set_xlabel("Intensity (" + self.intensity_unit + ")")
        axis.set_ylabel("Impact (%)")
        axis.set_title(title)
        axis.plot(self.intensity, self.mdd * 100, "b", label="MDD", **kwargs)
        axis.plot(self.intensity, self.paa * 100, "r", label="PAA", **kwargs)
        axis.plot(
            self.intensity, self.mdd * self.paa * 100, "k--", label="MDR", **kwargs
        )

        axis.set_xlim((self.intensity.min(), self.intensity.max()))
        axis.legend()
        return axis

    def check(self):
        """Check consistent instance data.

        Raises
        ------
        ValueError
        """
        num_exp = len(self.intensity)
        u_check.size(num_exp, self.mdd, "ImpactFunc.mdd")
        u_check.size(num_exp, self.paa, "ImpactFunc.paa")

        if num_exp == 0:
            LOGGER.warning(
                "%s impact function with name '%s' (id=%s) has empty" " intensity.",
                self.haz_type,
                self.name,
                self.id,
            )
            return

    @classmethod
    def from_step_impf(
        cls,
        intensity: tuple[float, float, float],
        haz_type: str,
        mdd: tuple[float, float] = (0, 1),
        paa: tuple[float, float] = (1, 1),
        impf_id: int = 1,
        **kwargs
    ):
        """Step function type impact function.

        By default, the impact is 100% above the step.
        Useful for high resolution modelling.

        Parameters
        ----------
        intensity: tuple(float, float, float)
            tuple of 3-intensity numbers: (minimum, threshold, maximum)
        haz_type: str
            the reference string for the hazard (e.g., 'TC', 'RF', 'WS', ...)
        mdd: tuple(float, float)
            (min, max) mdd values. The default is (0, 1)
        paa: tuple(float, float)
            (min, max) paa values. The default is (1, 1)
        impf_id : int, optional, default=1
            impact function id
        kwargs :
            keyword arguments passed to ImpactFunc()

        Return
        ------
        impf : climada.entity.impact_funcs.ImpactFunc
            Step impact function

        """
        inten_min, threshold, inten_max = intensity
        intensity = np.array([inten_min, threshold, threshold, inten_max])
        paa_min, paa_max = paa
        paa = np.array([paa_min, paa_min, paa_max, paa_max])
        mdd_min, mdd_max = mdd
        mdd = np.array([mdd_min, mdd_min, mdd_max, mdd_max])

        return cls(
            haz_type=haz_type,
            id=impf_id,
            intensity=intensity,
            mdd=mdd,
            paa=paa,
            **kwargs
        )

    def set_step_impf(self, *args, **kwargs):
        """This function is deprecated, use ImpactFunc.from_step_impf instead."""
        LOGGER.warning(
            "The use of ImpactFunc.set_step_impf is deprecated."
            + "Use ImpactFunc.from_step_impf instead."
        )
        self.__dict__ = ImpactFunc.from_step_impf(*args, **kwargs).__dict__

    @classmethod
    def from_sigmoid_impf(
        cls,
        intensity: tuple[float, float, float],
        L: float,
        k: float,
        x0: float,
        haz_type: str,
        impf_id: int = 1,
        **kwargs
    ):
        r"""Sigmoid type impact function hinging on three parameter.

        This type of impact function is very flexible for any sort of study,
        hazard and resolution. The sigmoid is defined as:

        .. math::

           f(x) = \frac{L}{1+e^{-k(x-x0)}}

        For more information: https://en.wikipedia.org/wiki/Logistic_function

        This method modifies self (climada.entity.impact_funcs instance)
        by assining an id, intensity, mdd and paa to the impact function.

        Parameters
        ----------
        intensity : tuple(float, float, float)
            tuple of 3 intensity numbers along np.arange(min, max, step)
        L : float
            "top" of sigmoid
        k : float
            "slope" of sigmoid
        x0 : float
            intensity value where f(x)==L/2
        haz_type: str
            the reference string for the hazard (e.g., 'TC', 'RF', 'WS', ...)
        impf_id : int, optional, default=1
            impact function id
        kwargs :
            keyword arguments passed to ImpactFunc()

        Return
        ------
        impf : climada.entity.impact_funcs.ImpactFunc
            Sigmoid impact function
        """
        inten_min, inten_max, inten_step = intensity
        intensity = np.arange(inten_min, inten_max, inten_step)
        paa = np.ones(len(intensity))
        mdd = L / (1 + np.exp(-k * (intensity - x0)))

        return cls(
            haz_type=haz_type,
            id=impf_id,
            intensity=intensity,
            paa=paa,
            mdd=mdd,
            **kwargs
        )

    def set_sigmoid_impf(self, *args, **kwargs):
        """This function is deprecated, use LitPop.from_countries instead."""
        LOGGER.warning(
            "The use of ImpactFunc.set_sigmoid_impf is deprecated."
            "Use ImpactFunc.from_sigmoid_impf instead."
        )
        self.__dict__ = ImpactFunc.from_sigmoid_impf(*args, **kwargs).__dict__

    @classmethod
    def from_poly_s_shape(
        cls,
        intensity: tuple[float, float, int],
        threshold: float,
        half_point: float,
        scale: float,
        exponent: float,
        haz_type: str,
        impf_id: int = 1,
        **kwargs
    ):
        r"""S-shape polynomial impact function hinging on four parameter.

        .. math::

            f(I) = \frac{\textrm{luk}(I)^{\textrm{exponent}}}{
                1 + \textrm{luk}(I)^{\textrm{exponent}}
            }
                \cdot \textrm{scale} \\
            \textrm{luk}(I) = \frac{\max[I - \textrm{threshold}, 0]}{
                \textrm{half_point} - \textrm{threshold}
            }

        This function is inspired by Emanuel et al. (2011)
        https://doi.org/10.1175/WCAS-D-11-00007.1

        This method only specifies mdd, and paa = 1 for all intensities.

        Parameters
        ----------
        intensity : tuple(float, float, float)
            tuple of 3 intensity numbers along np.linsapce(min, max, num)
        threshold : float
            Intensity threshold below which there is no impact.
            In general choose threshold > 0 for computational efficiency
            of impacts.
        half_point : float
            Intensity at which 50% of maximum impact is expected.
            If half_point <= threshold, mdd = 0 (and f(I)=0) for all
            intensities.
        scale : float
            Multiplicative factor for the whole function. Typically,
            this sets the maximum value at large intensities.
        exponent: float
            Exponent of the polynomial. Value must be exponent >= 0.
            Emanuel et al. (2011) uses the value 3.
        haz_type: str
            Reference string for the hazard (e.g., 'TC', 'RF', 'WS', ...)
        impf_id : int, optional, default=1
            Impact function id
        kwargs :
            keyword arguments passed to ImpactFunc()

        Raises
        ------
        ValueError : if exponent <= 0

        Returns
        -------
        impf : climada.entity.impact_funcs.ImpactFunc
            s-shaped polynomial impact function
        """
        if exponent < 0:
            raise ValueError("Exponent value must larger than 0")

        inten = np.linspace(*intensity)

        if threshold >= half_point:
            mdd = np.zeros_like(inten)
        else:
            luk = (inten - threshold) / (half_point - threshold)
            luk[luk < 0] = 0
            mdd = scale * luk**exponent / (1 + luk**exponent)
        paa = np.ones_like(inten)

        impf = cls(
            haz_type=haz_type, id=impf_id, intensity=inten, paa=paa, mdd=mdd, **kwargs
        )
        return impf

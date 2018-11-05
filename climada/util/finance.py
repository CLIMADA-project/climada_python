"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Finance functionalities.
"""
import logging

LOGGER = logging.getLogger(__name__)

def net_present_value(years, disc_rates, val_years):
    """Compute net present value.

    Parameters:
        years (np.array): array with the sequence of years to consider.
        disc_rates (np.array): discount rate for every year in years.
        val_years (np.array): chash flow at each year.

    Returns:
        float
    """
    if years.size != disc_rates.size or years.size != val_years.size:
        LOGGER.error('Wrong input sizes %s, %s, %s.', years.size,
                     disc_rates.size, val_years.size)
        raise ValueError

    npv = val_years[-1]
    for val, disc in zip(val_years[-2::-1], disc_rates[-2::-1]):
        npv = val + npv/(1+disc)

    return npv

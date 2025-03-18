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

This modules implements the Snapshot and SnapshotsCollection classes.

"""

import itertools
import logging
from dataclasses import dataclass
from weakref import WeakValueDictionary

from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.hazard import Hazard

LOGGER = logging.getLogger(__name__)


# TODO: Improve and make it an __eq__ function within Hazard?
def hazard_data_equal(haz1: Hazard, haz2: Hazard) -> bool:
    intensity_eq = (haz1.intensity != haz2.intensity).nnz == 0
    freq_eq = (haz1.frequency == haz2.frequency).all()
    frac_eq = (haz1.fraction != haz2.fraction).nnz == 0
    return intensity_eq and freq_eq and frac_eq


# TODO: Better measure checking (if we change the measure object after, changes are not accounted for!!!)
@dataclass(eq=False, frozen=True)
class Snapshot:
    """
    A snapshot of exposure, hazard, and impact function set for a given year.

    Attributes
    ----------
    exposure : Exposures
        Exposure data for the snapshot.
    hazard : Hazard
        Hazard data for the snapshot.
    impfset : ImpactFuncSet
        Impact function set associated with the snapshot.
    year : int
        Year of the snapshot.
    """

    exposure: Exposures
    hazard: Hazard
    impfset: ImpactFuncSet
    year: int
    measure: None | Measure = None

    # Class-level cache
    _instances = WeakValueDictionary()

    def __new__(cls, exposure, hazard, impfset, year, measure=None):
        """Check if an equal instance exists before creating a new one."""
        for existing_snapshot in cls._instances.values():
            if (
                existing_snapshot.exposure.gdf.equals(exposure.gdf)
                and hazard_data_equal(existing_snapshot.hazard, hazard)
                and existing_snapshot.impfset == impfset
                and existing_snapshot.year == year
            ):
                if (
                    existing_snapshot.measure
                    and measure
                    and existing_snapshot.measure.name == measure.name
                ):
                    LOGGER.debug(
                        f"Found existing instance of snapshot for year {year}, measure {measure.name}, with id {id(existing_snapshot)}"
                    )
                    return existing_snapshot  # Return existing instance
                elif existing_snapshot.measure is None and measure is None:
                    LOGGER.debug(
                        f"Found existing instance of snapshot for year {year} (no measure), with id {id(existing_snapshot)}"
                    )
                    return existing_snapshot  # Return existing instance

        # Create new instance if no match is found
        instance = super().__new__(cls)
        return instance

    def __post_init__(self):
        """Store the instance in the cache after initialization."""
        if id(self) not in self._instances:
            LOGGER.debug(f"Created and stored new Snapshot {id(self)}")
            self._instances[id(self)] = self

    def __eq__(self, value, /) -> bool:
        if not isinstance(value, Snapshot):
            return False
        if self is value:
            return True
        same_exposure = self.exposure.gdf.equals(value.exposure.gdf)
        same_hazard = hazard_data_equal(self.hazard, value.hazard)
        same_impfset = self.impfset == value.impfset
        same_year = self.year == value.year
        same_measure = self.measure == value.measure
        return (
            same_exposure
            and same_hazard
            and same_impfset
            and same_year
            and same_measure
        )

    def apply_measure(self, measure: Measure):
        LOGGER.debug(f"Applying measure {measure.name} on snapshot {id(self)}")
        exp_new, impfset_new, haz_new = measure.apply(
            self.exposure, self.impfset, self.hazard
        )
        return Snapshot(exp_new, haz_new, impfset_new, self.year, measure)


class SnapshotsCollection:
    """
    Collection of snapshots for different years.

    Attributes
    ----------
    exposure_set : dict
        Dictionary of exposure data by year.
    hazard_set : dict
        Dictionary of hazard data by year.
    impfset : ImpactFuncSet
        Impact function set shared across snapshots.
    snapshots_years : list of int
        Years associated with each snapshot in the collection.
    data : list of Snapshot
        List of Snapshot objects in the collection.
    """

    def __init__(self, snaplist):

        self._snapshots = {snap.year: snap for snap in snaplist}
        self._impfset = snaplist[0].impfset

    @classmethod
    def _from_dicts(
        cls,
        exposure_set: dict[int, Exposures],
        hazard_set: dict[int, Hazard],
        impfset: ImpactFuncSet,
        snapshot_years: list[int],
    ):

        # Validate all requested years exist
        missing_exposure = [y for y in snapshot_years if y not in exposure_set]
        missing_hazard = [y for y in snapshot_years if y not in hazard_set]
        if missing_exposure or missing_hazard:
            raise ValueError(
                f"Missing data for years - Exposure: {missing_exposure}, Hazard: {missing_hazard}"
            )

        return cls(
            [
                Snapshot(exposure_set[year], hazard_set[year], impfset, year)
                for year in sorted(snapshot_years)
            ]
        )

    @property
    def data(self):
        return list(self._snapshots.values())

    @property
    def snapshots_years(self):
        return self._snapshots.keys()

    @property
    def exposure_set(self):
        return [snap.exposure for snap in self._snapshots.values()]

    @property
    def hazard_set(self):
        return [snap.hazard for snap in self._snapshots.values()]

    @property
    def impfset(self):
        return self._impfset

    def __len__(self):
        """Return the number of snapshots in the collection."""
        return len(self._snapshots)

    # def __iter__(self):
    #     """Return an iterator over the snapshots in the collection."""
    #     return iter(self.data)

    def __contains__(self, item):
        """Check if a Snapshot or a year exists in the collection."""
        if isinstance(item, int):
            return item in self._snapshots
        if isinstance(item, Snapshot):
            return item in self._snapshots.values()  # Check object identity
        else:
            return False  # Invalid type

    # Check that at least first and last snap are complete
    # and otherwise it is ok

    @classmethod
    def from_dict(cls, snapshots_dict, impfset):
        """
        Create a SnapshotsCollection from a dictionary of snapshots.

        Parameters
        ----------
        snapshots_dict : dict
            Dictionary of snapshots data by year.
        impfset : ImpactFuncSet
            Impact function set shared across snapshots.

        Returns
        -------
        SnapshotsCollection
            A new SnapshotsCollection instance.
        """
        snapshot_years = list(snapshots_dict.keys())
        exposure_set = {year: snapshots_dict[year][0] for year in snapshot_years}
        hazard_set = {year: snapshots_dict[year][1] for year in snapshot_years}
        return cls._from_dicts(
            exposure_set=exposure_set,
            hazard_set=hazard_set,
            impfset=impfset,
            snapshot_years=snapshot_years,
        )

    @classmethod
    def from_lists(cls, hazard_list, exposure_list, impfset, snapshot_years):
        """
        Create a SnapshotsCollection from separate lists of hazard and exposure data.

        Parameters
        ----------
        hazard_list : list
            List of hazard data for each year, in the same order as `snapshot_years`.
        exposure_list : list
            List of exposure data for each year, in the same order as `snapshot_years`.
        impfset : ImpactFuncSet
            Impact function set shared across snapshots.
        snapshot_years : list of int
            List of years corresponding to each hazard and exposure data entry.

        Returns
        -------
        SnapshotsCollection
            A new SnapshotsCollection instance.
        """
        exposure_set = {year: exposure_list[i] for i, year in enumerate(snapshot_years)}
        hazard_set = {year: hazard_list[i] for i, year in enumerate(snapshot_years)}
        return cls._from_dicts(
            exposure_set=exposure_set,
            hazard_set=hazard_set,
            impfset=impfset,
            snapshot_years=snapshot_years,
        )

    def add_snapshot(self, snapshot: Snapshot):
        """Adds a snapshot to the collecton"""
        if not isinstance(snapshot, Snapshot):
            raise TypeError("snapshot must be an instance of Snapshot")

        if snapshot in self:  # Identity check
            LOGGER.warning("Snapshot already present.", UserWarning)
            return  # Do nothing if it's the exact same object

        if snapshot.year in self:
            LOGGER.warning(
                "Snapshot already exist for this year. Overwriting.", UserWarning
            )

        # Ensure the impact function set is consistent
        if snapshot.impfset is not self.impfset:
            raise ValueError(
                "Snapshot impact function set does not match existing one."
            )

        self._snapshots[snapshot.year] = snapshot
        self._snapshots = dict(sorted(self._snapshots.items()))

    def pairwise(self):
        """
        Generate pairs of successive elements from an iterable.

        Parameters
        ----------
        iterable : iterable
            An iterable sequence from which successive pairs of elements are generated.

        Returns
        -------
        zip
            A zip object containing tuples of successive pairs from the input iterable.

        Example
        -------
        >>> list(pairwise([1, 2, 3, 4]))
        [(1, 2), (2, 3), (3, 4)]
        """
        a, b = itertools.tee(self._snapshots.values())
        next(b, None)
        return zip(a, b)

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

"""

import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Set, Union

from climada.engine.option_appraisal.MCDM.constants import (
    DEFAULT_CATEGORY_WEIGHT,
    IMPORTANCE_MATCH,
)
from climada.engine.option_appraisal.MCDM.weights import WeightedItem

# Define type aliases for cleaner type hints
CategoryName = str
CategoryLike = Union[CategoryName, "CriteriaCategory"]

LOGGER = logging.getLogger(__name__)


class CategorySpace:
    """Manages a dedicated, isolated registry of CriteriaCategory objects."""

    # Class attribute to hold the singleton default space
    _default_space: Optional["CategorySpace"] = None

    def __init__(self):
        self._registry: Dict[str, "CriteriaCategory"] = {}
        self.category_weights = None

    # --- New Class Method for Default Access ---
    @classmethod
    def get_default_space(cls) -> "CategorySpace":
        """Returns the default CategorySpace instance, creating it if necessary."""
        if cls._default_space is None:
            cls._default_space = CategorySpace()
        return cls._default_space

    def reset_categories(self):
        self._registry.clear()

    def get(self, name: CategoryName) -> Optional["CriteriaCategory"]:
        """
        Retrieves a criteria category object by its unique name.

        Parameters
        ----------
        name : CategoryName
            The unique name of the criteria category.

        Returns
        -------
        Optional[CriteriaCategory]
            The CriteriaCategory object if found, otherwise None.
        """
        return self._registry.get(name)

    def __contains__(self, item: object) -> bool:
        """
        Allows checking if a category is in this space using 'category in space'.

        It checks if the item is a CriteriaCategory object and if it is
        present in this space's internal registry.
        """
        if not isinstance(item, CriteriaCategory):
            return False

        return item.name in self._registry

    def remove(self, category: CategoryLike) -> None:
        if not isinstance(category, str):
            try:
                category = category.name
            except AttributeError as err:
                err.add_note(
                    "category has to be the name of category or a Category object."
                )
                raise

        cat = self._registry[category]
        for parent in cat.parents:
            parent._children.remove(cat)
            parent._children.update(cat.children)

        for child in cat.children:
            child._parents.remove(cat)
            child._parents.update(cat.parents)

        del self._registry[category]

    def register(self, category: "CriteriaCategory") -> None:
        if category.name in self._registry:
            raise ValueError(
                f"Category '{category.name}' already exists in space '{self.name}'."
            )
        self._registry[category.name] = category

    def add_category(
        self,
        name: CategoryName,
        parent_cats: Optional[Union[CategoryLike, List[CategoryLike]]] = None,
        category_type: Optional[str] = None,
        overwrite: bool = False,
    ) -> "CriteriaCategory":
        return CriteriaCategory(
            name,
            parents=parent_cats,
            category_type=category_type,
            overwrite=overwrite,
            space=self,
        )

    def select_categories_by_type(
        self, category_types: str | Iterable[str]
    ) -> "list[CriteriaCategory]":
        if isinstance(category_types, str):
            category_types = [category_types]

        return [
            category
            for category in self.all_categories
            if category.category_type and category.category_type in category_types
        ]

    def create_subset_by_type(
        self, category_types: str | Iterable[str]
    ) -> "CategorySpace":
        if isinstance(category_types, str):
            category_types = [category_types]

        subspace = CategorySpace()
        selected_categories = self.select_categories_by_type(category_types)
        for category in selected_categories:
            subspace.register(category)

        return subspace

    def create_subspace(
        self, selection: Union[CategoryLike, List[CategoryLike]]
    ) -> "CategorySpace":
        if not isinstance(selection, Iterable):
            selection = [selection]

        subspace = CategorySpace()
        for category in selection:
            if isinstance(category, str):
                category = self.get(category)

            subspace.register(category)

        return subspace

    @property
    def category_weights(self):
        return {k: self._registry[k].weight for k in self._registry.keys()}

    @category_weights.setter
    def category_weights(self, value, /):
        if value is None:
            for k, v in {
                cat_name: DEFAULT_CATEGORY_WEIGHT for cat_name in self._registry.keys()
            }.items():
                self._registry[k].weight = v
        else:
            no_match = [k for k in value.keys() if k not in self._registry.keys()]
            no_weight = [k for k in self._registry.keys() if k not in value.keys()]
            if len(no_match) > 0:
                LOGGER.warning(
                    f"Some weights do not correspond to any category: {no_match}"
                )

            if len(no_weight) > 0:
                LOGGER.warning(
                    f"No weight given for one or more categories: {no_weight}\n(will use existing or default ({DEFAULT_CATEGORY_WEIGHT}))"
                )

            for k, v in value.items():
                self._registry[k].weight = v

    @property
    def all_categories(self) -> List["CriteriaCategory"]:
        return list(self._registry.values())

    @property
    def category_types(self):
        return list(set([cat.category_type for cat in self.all_categories]))

    def display(self) -> None:
        """
        Prints the entire category hierarchy registered in the system using ASCII art.
        It handles multiple roots and is robust against multiple inheritance.
        """
        if not self._registry:
            print("The category registry is empty.")
            return

        # 1. Identify all root nodes (categories with no parents)
        # Note: In a pure hierarchy, this is simple. With multiple inheritance (DAG),
        # a category can be a root even if it has parents *not* in the registry,
        # but here we assume all parents are created via the provided methods.
        root_nodes = sorted(
            [category for category in self._registry.values() if not category.parents],
            key=lambda c: c.name,
        )  # Sort by name for stable output

        if not root_nodes:
            # This can happen in a pure graph/cyclic structure, or if only children were defined.
            print("No category without a parent was found to act as a root.")
            print(f"Categories present: {list(self._registry.keys())}")
            return

        print("\n--- Criteria Category Hierarchy ---")

        # 2. Use a recursive helper function to print the tree starting from roots
        def print_node_recursive(
            node: CriteriaCategory, prefix: str = "", is_last: bool = True
        ) -> None:
            """Recursively prints the node and its children."""

            # ASCII art characters
            connector = "└── " if is_last else "├── "

            # Print the current node
            print(
                prefix
                + connector
                + node.category_type
                + ": "
                + node.name
                + " category weight: "
                + str(self.category_weights[node.name])
            )

            # Determine the prefix for the children
            # If the current node is the last child of its parent, its children's prefix
            # uses a space/indent. Otherwise, it uses the vertical line.
            child_prefix = prefix + ("    " if is_last else "│   ")

            # Sort children by name for predictable display order
            sorted_children = sorted(list(node.children), key=lambda c: c.name)

            # Recursively call for children
            for i, child in enumerate(sorted_children):
                is_last_child = i == len(sorted_children) - 1
                print_node_recursive(child, child_prefix, is_last_child)

        # Print each root
        for i, root in enumerate(root_nodes):
            is_last_root = i == len(root_nodes) - 1
            # Root nodes use slightly different logic for the final block
            print_node_recursive(root, "", is_last_root)
            if not is_last_root:
                # Add a blank line between separate root trees for clarity
                print()


class CriteriaCategory(WeightedItem):
    """
    Represents a criteria category in a dynamic, multiple-parent hierarchy.

    This class manages the structure and relationships of criteria, allowing
    for runtime definition and retrieval of categories. It implements logic
    to check for ancestral relationships, respecting multiple parent links.

    Attributes
    ----------
    _registry : dict[CategoryName, CriteriaCategory]
        A class-level dictionary serving as a global lookup for all defined
        CriteriaCategory instances, keyed by their name.
    name : CategoryName
        The unique name of the criteria category.
    parents : set[CriteriaCategory]
        The set of direct parent categories this criteria inherits from.
    children : set[CriteriaCategory]
        The set of direct child categories that inherit from this criteria.
    """

    _registry: dict[CategoryName, "CriteriaCategory"] = {}

    def __init__(
        self,
        name: CategoryName,
        parents: Optional[Union[CategoryLike, List[CategoryLike]]] = None,
        category_type: Optional[str] = None,
        category_weight: Optional[float] = None,
        space: Optional[CategorySpace] = None,
        overwrite: Optional[bool] = False,
    ) -> None:
        """
        Initializes a new CriteriaCategory.

        Parameters
        ----------
        name : CategoryName
            The unique name of the category.
        parents : Optional[Union[CategoryLike, List[CategoryLike]]], optional
            The parent criteria(s) this category inherits from. Can be a single
            name/object or a list of names/objects. By default, None.

        Raises
        ------
        ValueError
            If a category with the given name already exists in the registry.
        """
        WeightedItem.__init__(self, category_weight)
        self._space = space if space is not None else CategorySpace.get_default_space()
        self.name: CategoryName = name
        self.category_type: str | None = category_type
        self._parents: Set[CriteriaCategory] = set()
        self._children: Set[CriteriaCategory] = set()

        if self in self.space:
            if not overwrite:
                if not self.space.get(name).has_parents_exactly(parents):
                    raise ValueError(
                        f"CriteriaCategory '{name}' with different parents ({self.space.get(name)._parents_name} != {parents}) already exists in current category space ({self.space}). You can overwrite with `overwrite=True`."
                    )
                return

        self.space.register(self)
        if parents:
            self._add_parents(parents)

    @property
    def space(self):
        return self._space

    @property
    def parents(self):
        return self._parents

    @property
    def children(self):
        return self._children

    def __eq__(self, other: Any) -> bool:
        """
        Defines equality based solely on the category name.
        """
        # 1. Check if the other object is an instance of CriteriaCategory
        if not isinstance(other, CriteriaCategory):
            return NotImplemented  # Defer to the other object's __eq__

        # 2. Compare names
        return self.name == other.name

    def __hash__(self) -> int:
        """
        Defines the hash based solely on the category name.
        Required for objects used in sets or as dictionary keys.
        """
        return hash(self.name)

    @property
    def _parents_names(self):
        return [p.name for p in self.parents]

    def has_parents_exactly(
        self, check_parents: Union[CategoryLike, List[CategoryLike], None]
    ) -> bool:
        """
        Checks if the criteria's set of parents is exactly equal to the provided set of parents.

        This check is order-independent.

        Parameters
        ----------
        check_parents : Union[CategoryLike, List[CategoryLike]]
            A single parent or a list of parent names or CriteriaCategory objects
            to compare against the criteria's actual parents.

        Returns
        -------
        bool
            True if the provided list of parents (after resolution) is exactly
            the same set as the criteria's actual parents, False otherwise.

        Raises
        ------
        ValueError
            If any parent name in `check_parents` cannot be found in tself.parents == resolved_check_parentshe registry.
        """
        if check_parents is None and self.parents == set():
            return True

        if not isinstance(check_parents, list):
            check_parents = [check_parents]

        return check_parents == self._parents_names

    def _add_parents(
        self, parent_names: Union[CategoryLike, List[CategoryLike]]
    ) -> None:
        """
        Internal helper to resolve and establish parent links.

        Parameters
        ----------
        parent_names : Union[CategoryLike, List[CategoryLike]]
            The parent criteria(s) to link.

        Raises
        ------
        ValueError
            If any parent category specified by name is not found.
        TypeError
            If an item in the parent list is not a string or CriteriaCategory object.
        """
        if not isinstance(parent_names, list):
            parent_names = [parent_names]

        for p_name in parent_names:
            parent_obj: Optional[CriteriaCategory] = None

            if isinstance(p_name, str):
                parent_obj = self.space.get(p_name)
                if not parent_obj:
                    raise ValueError(f"Parent criteria '{p_name}' not found.")
            elif isinstance(p_name, CriteriaCategory):
                parent_obj = p_name
            else:
                raise TypeError(
                    "Parents must be a string (category name) or a CriteriaCategory object."
                )

            self.parents.add(parent_obj)
            parent_obj.children.add(self)

    def is_a(self, other_category: CategoryLike | None) -> bool:
        """
        Checks if this criteria is a subcategory (descendant) of or is
        the target category.

        It performs a Breadth-First Search (BFS) up the parent hierarchy
        to account for multiple parent links.

        Parameters
        ----------
        other_category : CategoryLike
            The target category to check against. Can be its name or object.

        Returns
        -------
        bool
            True if this category is a descendant of or is the target category,
            False otherwise.

        Notes
        -----
        Uses BFS with a visited set to handle cycles that might exist in
        complex, manually defined DAGs, ensuring termination.
        """
        if other_category is None:
            return False

        if isinstance(other_category, str):
            other_category = self.space.get(other_category)
            if not other_category:
                return False

        # BFS approach to traverse multiple parents
        visited: Set[CriteriaCategory] = set()
        to_visit: List[CriteriaCategory] = [self]

        while to_visit:
            current = to_visit.pop(0)

            if current is other_category:
                return True

            if current not in visited:
                visited.add(current)
                to_visit.extend(list(current.parents))  # Add all parents to the queue

        return False

    def __repr__(self, indent=0) -> str:
        parent_names = sorted([p.name for p in self.parents])
        parent_str = f" Parents: {', '.join(parent_names)}" if parent_names else "none"
        indent_space = " " * indent
        return f"""{indent_space}name: {self.name} weight: {self.weight} type: {self.category_type}\n{indent_space}parents: {parent_str}"""


def create_criteria_category(
    name: CategoryName,
    parent_cats: Optional[Union[CategoryLike, List[CategoryLike]]] = None,
    category_type: Optional[str] = None,
    space: Optional[CategorySpace] = None,
    overwrite: bool = False,
) -> CriteriaCategory:
    """
    Convenience function to simplify dynamic creation of CriteriaCategory objects.

    Parameters
    ----------
    name : CategoryName
        The unique name of the new criteria category.
    parent_names : Optional[Union[CategoryName, List[CategoryName]]], optional
        The name(s) of the parent criteria. By default, None.

    Returns
    -------
    CriteriaCategory
        The newly created criteria category object.
    """
    return CriteriaCategory(
        name,
        parents=parent_cats,
        category_type=category_type,
        space=space,
        overwrite=overwrite,
    )


def update_categories_from_dict(
    hierarchy_dict: Dict[str, Any], space: Optional[CategorySpace] = None
) -> None:
    """
    Updates or creates the internal hierarchy of CriteriaCategory objects from a nested dictionary.

    The keys of the dictionary become the new categories, and their immediate parents
    are passed down through the recursion.

    Parameters
    ----------
    hierarchy_dict : Dict[str, Any]
        The dictionary representing the hierarchy. Keys are category names.
        Values can be another nested dictionary (representing children) or None/Empty dict.

    Raises
    ------
    TypeError
        If a value in the dictionary is neither a dictionary nor None.
    ValueError
        If a category name is non-unique (already exists).

    Notes
    -----
    The function modifies the global CriteriaCategory._registry as a side effect.
    """
    space = CategorySpace.get_default_space() if space is None else space
    return __categories_hierarchy_recursion(hierarchy_dict, space)


def __categories_hierarchy_recursion(
    hierarchy_dict: Dict[str, Any],
    space: CategorySpace,
    current_parents: Optional[Union[CategoryLike, List[CategoryLike]]] = None,
) -> None:
    """
    Recursively creates a hierarchy of CriteriaCategory objects from a nested dictionary.

    The keys of the dictionary become the new categories, and their immediate parents
    are passed down through the recursion.

    Parameters
    ----------
    hierarchy_dict : Dict[str, Any]
        The dictionary representing the hierarchy. Keys are category names.
        Values can be another nested dictionary (representing children) or None/Empty dict.
    current_parents : Optional[Union[str, List[str]]], optional
        The name(s) of the categories that should be set as the parent(s) for the
        current level's keys. Used internally for recursion. By default, None.

    Raises
    ------
    TypeError
        If a value in the dictionary is neither a dictionary nor None.
    ValueError
        If a category name is non-unique (already exists).

    Notes
    -----
    The function modifies the global CriteriaCategory._registry as a side effect.
    """

    # Ensure current_parents is always a list for consistent handling
    if current_parents is None:
        parent_list = []
    elif isinstance(current_parents, str):
        parent_list = [current_parents]
    else:
        parent_list = current_parents

    for category_name, value in hierarchy_dict.items():
        try:
            create_criteria_category(
                name=category_name, parent_cats=parent_list, space=space
            )
        except ValueError as e:
            if "different parents" in str(e):
                space.get(category_name)._add_parents(parent_list)
            else:
                print(
                    f"Warning: Category '{category_name}' skipped (likely duplicate). Error: {e}"
                )
            continue

        # Recurse if there are children
        if value is None:
            continue

        if isinstance(value, dict):
            new_parent_for_children: Union[str, List[str]] = category_name
            __categories_hierarchy_recursion(value, space, new_parent_for_children)

        elif not isinstance(value, dict) and value is not None:
            raise TypeError(
                f"Value for category '{category_name}' must be a dictionary (for children) or None, "
                f"but got {type(value).__name__}."
            )


class CategorizedObject:
    """
    An object that uses composition to belong to one or more CriteriaCategories.

    The object maintains a set of direct criteria links. Checks against the
    hierarchy are delegated to the CriteriaCategory system.

    Attributes
    ----------
    name : str
        The name or identifier of the object.
    categories : set[CriteriaCategory]
        The set of CriteriaCategory objects this instance is directly assigned to.
    """

    def __init__(
        self,
        name: str,
        categories: Optional[
            Union[CategoryLike, List[CategoryLike], Set[CriteriaCategory]]
        ] = None,
        space: Optional[CategorySpace] = None,
    ) -> None:
        """
        Initializes the CategorizedObject.

        Parameters
        ----------
        name : str
            The name or identifier of the object.
        categories : Optional[Union[CategoryName, List[CategoryName]]], optional
            The name(s) of the initial categories to assign. By default, None.
        """
        self.name: str = name
        self._categories: Set[CriteriaCategory] = set()
        self._space = space if space is not None else CategorySpace.get_default_space()
        if categories:
            self.add_categories(categories)

    @property
    def space(self):
        return self._space

    @property
    def categories(self):
        return self._categories

    @property
    def category_space(self):
        return self._space

    def add_categories(
        self, categories: Union[CategoryLike, List[CategoryLike], Set[CriteriaCategory]]
    ) -> None:
        """
        Adds one or more criteria categories to the object by name.

        Parameters
        ----------
        category_names : Union[CategoryName, List[CategoryName]]
            The name(s) of the criteria categories to add.

        Raises
        ------
        ValueError
            If a category name does not exist in the CriteriaCategory registry.
        """
        if not isinstance(categories, (list, set)):
            categories = [categories]

        for cat in categories:
            if not isinstance(cat, CriteriaCategory):
                cat = self.category_space.get(cat)
            if not cat:
                # Enforce that categories must be defined globally before being assigned to an object
                raise ValueError(
                    f"CriteriaCategory '{cat}' is not defined. Create it first with `CriteriaCategory.create_criteria_category()`"
                )
            self.categories.add(cat)

    def has_category(self, category_name: CategoryLike) -> bool:
        """
        Checks if the object belongs to the specified category or any of its
        subcategories in the hierarchy.

        Parameters
        ----------
        category_name : CategoryName
            The name of the criteria category to check against.

        Returns
        -------
        bool
            True if the object is directly or indirectly a member of the
            target category, False otherwise.
        """
        if isinstance(category_name, str):
            target_category = self.category_space.get(category_name)
            if not target_category:
                return False

        elif isinstance(category_name, CriteriaCategory):
            target_category = category_name
        else:
            raise ValueError(f"{category_name} is not a string or a CriteriaCategory")

        for category in self.categories:
            if category.is_a(target_category):
                return True
        return False

    def __repr__(self) -> str:
        cat_names = sorted([c.name for c in self.categories])
        return f"<CategorizedObject: {self.name} | Criteria: {', '.join(cat_names)}>"

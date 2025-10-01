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

from typing import Any, Dict, List, Optional, Set, Union

from numpy import False_

# Define type aliases for cleaner type hints
CategoryName = str
CategoryLike = Union[CategoryName, "CriteriaCategory"]


class CriteriaCategory:
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
        self.name: CategoryName = name
        self.parents: Set[CriteriaCategory] = set()
        self.children: Set[CriteriaCategory] = set()

        if not overwrite:
            if name in CriteriaCategory._registry:
                if not CriteriaCategory._registry[name].has_parents_exactly(parents):
                    raise ValueError(
                        f"CriteriaCategory '{name}' with different parents ({CriteriaCategory._registry[name]._parents_names} != {parents}) already exists. You can overwrite with `overwrite=True`."
                    )

        CriteriaCategory._registry[name] = self
        if parents:
            self._add_parents(parents)

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

    @classmethod
    def reset_categories(cls):
        cls._registry = {}

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
                parent_obj = CriteriaCategory._registry.get(p_name)
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
            other_category = CriteriaCategory._registry.get(other_category)
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

    @classmethod
    def get(cls, name: CategoryName) -> Optional["CriteriaCategory"]:
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
        return cls._registry.get(name)

    def __repr__(self) -> str:
        parent_names = sorted([p.name for p in self.parents])
        parent_str = f" Parents: {', '.join(parent_names)}" if parent_names else ""
        return f"<CriteriaCategory: {self.name}{parent_str}>"

    @classmethod
    def display(cls) -> None:
        """
        Prints the entire category hierarchy registered in the system using ASCII art.
        It handles multiple roots and is robust against multiple inheritance.
        """
        if not cls._registry:
            print("The category registry is empty.")
            return

        # 1. Identify all root nodes (categories with no parents)
        # Note: In a pure hierarchy, this is simple. With multiple inheritance (DAG),
        # a category can be a root even if it has parents *not* in the registry,
        # but here we assume all parents are created via the provided methods.
        root_nodes = sorted(
            [category for category in cls._registry.values() if not category.parents],
            key=lambda c: c.name,
        )  # Sort by name for stable output

        if not root_nodes:
            # This can happen in a pure graph/cyclic structure, or if only children were defined.
            print("No category without a parent was found to act as a root.")
            print(f"Categories present: {list(cls._registry.keys())}")
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
            print(prefix + connector + node.name)

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


def create_criteria_category(
    name: CategoryName,
    parent_cats: Optional[Union[CategoryLike, List[CategoryLike]]] = None,
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
    return CriteriaCategory(name, parents=parent_cats, overwrite=overwrite)


def update_categories_from_dict(hierarchy_dict: Dict[str, Any]) -> None:
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
    return __categories_hierarchy_recursion(hierarchy_dict)


def __categories_hierarchy_recursion(
    hierarchy_dict: Dict[str, Any],
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
            create_criteria_category(name=category_name, parent_cats=parent_list)
        except ValueError as e:
            if "different parents" in str(e):
                CriteriaCategory.get(category_name)._add_parents(parent_list)
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
            __categories_hierarchy_recursion(value, new_parent_for_children)

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
        self.categories: Set[CriteriaCategory] = set()
        if categories:
            self.add_categories(categories)

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
                cat = CriteriaCategory.get(cat)
            if not cat:
                # Enforce that categories must be defined globally before being assigned to an object
                raise ValueError(
                    f"CriteriaCategory '{cat}' is not defined. Create it first with `CriteriaCategory.create_criteria_category()`"
                )
            self.categories.add(cat)

    def has_category(self, category_name: CategoryName) -> bool:
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
        target_category = CriteriaCategory.get(category_name)
        if not target_category:
            return False

        for category in self.categories:
            if category.is_a(target_category):
                return True
        return False

    def __repr__(self) -> str:
        cat_names = sorted([c.name for c in self.categories])
        return f"<CategorizedObject: {self.name} | Criteria: {', '.join(cat_names)}>"

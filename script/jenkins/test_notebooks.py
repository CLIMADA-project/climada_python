#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import unittest
from pathlib import Path

import nbformat
import pytest

BOUND_TO_FAIL = "# Note: execution of this cell will fail"
"""Cells containing this line will not be executed in the test"""

EXCLUDED_FROM_NOTEBOOK_TEST = ["climada_installation_step_by_step.ipynb"]
"""These notebooks are excluded from being tested"""


# collect test cases, one for each notebook in the docs (unless they're excluded)
NOTEBOOK_DIR = Path(__file__).parent.parent.parent.joinpath("doc", "tutorial")
NOTEBOOKS = [
    (f.absolute(), f.name)
    for f in sorted(NOTEBOOK_DIR.iterdir())
    if os.path.splitext(f)[1] == (".ipynb")
    and not f.name in EXCLUDED_FROM_NOTEBOOK_TEST
]


@pytest.mark.parametrize("nb, name", NOTEBOOKS)
def test_notebook(nb, name):
    """Extracts code cells from the notebook and executes them one by one, using `exec`.
    Magic lines and help/? calls are eliminated.
    Cells containing `BOUND_TO_FAIL` are elided.
    Cells doing multiprocessing are elided."""
    notebook = nb
    cwd = Path.cwd()
    try:
        # cd to the notebook directory
        os.chdir(notebook.absolute().parent)
        print(f"start testing {notebook}")

        # read the notebook into a string
        with open(notebook.name, encoding="utf8") as nb:
            content = nb.read()

        # parse the string with nbformat.reads
        cells = nbformat.reads(content, 4)["cells"]

        # create namespace with IPython standards
        namespace = dict()
        exec("from IPython.display import display", namespace)

        # run all cells
        i = 0
        for c in cells:

            # skip markdown cells
            if c["cell_type"] != "code":
                continue
            i += 1

            # skip deliberately failing cells
            if BOUND_TO_FAIL in c["source"]:
                continue

            # skip multiprocessing cells
            if any(
                [
                    tabu in c["source"].split()
                    for tabu in [
                        "import multiprocessing",
                        "from multiprocessing import",
                    ]
                ]
            ):
                print(
                    "\n".join(
                        [
                            f"\nskip multiprocessing cell {i} in {notebook.name}",
                            "+" + "-" * 68 + "+",
                            c["source"],
                        ]
                    )
                )
                continue

            # remove non python lines and help calls which require user input
            # or involve pools being opened/closed
            python_code = "\n".join(
                [
                    re.sub(r"pool=\w+", "pool=None", ln)
                    for ln in c["source"].split("\n")
                    if not ln.startswith("%")
                    and not ln.startswith("help(")
                    and not ln.startswith("ask_ok(")
                    and not ln.startswith("ask_ok(")
                    and not ln.startswith(
                        "pool"
                    )  # by convention Pool objects are called pool
                    and not ln.strip().endswith("?")
                    and not re.search(
                        r"(\W|^)Pool\(", ln
                    )  # prevent Pool object creation
                ]
            )

            # execute the python code
            try:
                exec(python_code, namespace)

            # report failures
            except Exception as e:
                failure = "\n".join(
                    [
                        f"notebook {notebook.name} cell {i} failed with {e.__class__}",
                        f"{e}",
                        "+" + "-" * 68 + "+",
                        c["source"],
                    ]
                )
                print(f"failed {notebook}")
                print(failure)
                raise e

        print(f"succeeded {notebook}")
    finally:
        os.chdir(cwd)


def main():
    # run the tests and write xml reports to tests_xml
    pytest.main([f"--junitxml=tests_xml/tests.xml", __file__])


if __name__ == "__main__":
    if sys.argv[1] == "report":
        main()

    else:
        nb = sys.argv[1]
        test_notebook(Path(nb), nb)

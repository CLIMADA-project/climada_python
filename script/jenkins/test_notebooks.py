#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import unittest
from pathlib import Path
import nbformat

import climada

NOTEBOOK_DIR = Path(__file__).parent.joinpath('doc', 'tutorial')
'''The path to the notebook directories.'''

BOUND_TO_FAIL = '# Note: execution of this cell will fail'
'''Cells containing this line will not be executed in the test'''

EXCLUDED_FROM_NOTEBOOK_TEST = ['climada_installation_step_by_step.ipynb']
'''These notebooks are excluded from being tested'''

class NotebookTest(unittest.TestCase):
    '''Generic TestCase for testing the executability of notebooks

    Attributes
    ----------
    wd : str
        Absolute Path to the working directory, i.e., the directory of the notebook.
    notebook : str
        File name of the notebook.

    '''

    def __init__(self, methodName, wd=None, notebook=None):
        super(NotebookTest, self).__init__(methodName)
        self.wd = wd
        self.notebook = notebook

    def test_notebook(self):
        '''Extracts code cells from the notebook and executes them one by one, using `exec`.
        Magic lines and help/? calls are eliminated.
        Cells containing `BOUND_TO_FAIL` are elided.
        Cells doing multiprocessing are elided.'''

        cwd = Path.cwd()
        try:
            # cd to the notebook directory
            os.chdir(self.wd)
            print(f'start testing {self.notebook}')

            # read the notebook into a string
            with open(self.notebook, encoding='utf8') as nb:
                content = nb.read()

            # parse the string with nbformat.reads
            cells = nbformat.reads(content, 4)['cells']

            # create namespace with IPython standards
            namespace = dict()
            exec('from IPython.display import display', namespace)

            # run all cells
            i = 0
            for c in cells:

                # skip markdown cells
                if c['cell_type'] != 'code': continue
                i += 1

                # skip deliberately failing cells
                if BOUND_TO_FAIL in c['source']: continue

                # skip multiprocessing cells
                if any([ tabu in c['source'].split() for tabu in [
                    'import multiprocessing',
                    'from multiprocessing import',
                ]]):
                    print('\n'.join([
                        f'\nskip multiprocessing cell {i} in {self.notebook}',
                        '+'+'-'*68+'+',
                        c['source']
                    ]))
                    continue

                # remove non python lines and help calls which require user input
                # or involve pools being opened/closed
                python_code = "\n".join([
                    re.sub(r'pool=\w+', 'pool=None', ln)
                    for ln in c['source'].split("\n")
                    if not ln.startswith('%')
                    and not ln.startswith('help(')
                    and not ln.startswith('ask_ok(')
                    and not ln.startswith('ask_ok(')
                    and not ln.startswith('pool')  # by convention Pool objects are called pool
                    and not ln.strip().endswith('?')
                    and not 'Pool(' in ln  # prevent Pool object creation
                ])

                # execute the python code
                try:
                    exec(python_code, namespace)

                # report failures
                except Exception as e:
                    failure = "\n".join([
                        f"notebook {self.notebook} cell {i} failed with {e.__class__}",
                        f"{e}",
                        '+'+'-'*68+'+',
                        c['source']
                    ])
                    print(f'failed {self.notebook}')
                    print(failure)
                    self.fail(failure)

            print(f'succeeded {self.notebook}')
        finally:
            os.chdir(cwd)


def main():
    # list notebooks in the NOTEBOOK_DIR
    notebooks = [f.absolute()
                 for f in sorted(NOTEBOOK_DIR.iterdir())
                 if os.path.splitext(f)[1] == ('.ipynb')
                 and not f.name in EXCLUDED_FROM_NOTEBOOK_TEST]

    # build a test suite with a test for each notebook
    suite = unittest.TestSuite()
    for notebook in notebooks:
        class NBTest(NotebookTest): pass
        test_name = "_".join(notebook.stem.split())
        setattr(NBTest, test_name, NBTest.test_notebook)
        suite.addTest(NBTest(test_name, notebook.parent, notebook.name))

    # run the tests depending on the first input argument: None or 'report'.
    # write xml reports for 'report'
    if sys.argv[1:]:
        arg = sys.argv[1]
        if arg == 'report':
            import xmlrunner
            outdirstr = str(Path(__file__).parent.joinpath('tests_xml'))
            xmlrunner.XMLTestRunner(output=outdirstr).run(suite)
        else:
            jd, nb = os.path.split(arg)
            unittest.TextTestRunner(verbosity=2).run(NotebookTest('test_notebook', jd, nb))
    # with no argument just run the test
    else:
        unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    sys.path.append(str(Path.cwd()))
    main()

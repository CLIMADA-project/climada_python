#!/usr/bin/env python
# coding: utf-8

import os
import sys
import unittest
import nbformat

from climada.util.constants import SOURCE_DIR


NOTEBOOK_DIR = os.path.abspath('doc/tutorial')
'''The path to the notebook directories.'''

BOUND_TO_FAIL = '# Note: execution of this cell will fail'
'''Cells containing this line will not be executed in the test'''


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

        # cd to the notebook directory
        os.chdir(self.wd)
        print(f'start testing {self.notebook}')

        # read the notebook into a string
        with open(self.notebook, encoding='utf8') as nb:
            content = nb.read()
        
        # parse the string with nbformat.reads
        cells = nbformat.reads(content, 4)['cells']
        
        namespace = dict()
        for i, c in enumerate(cells):
            
            # skip markdown cells
            if c['cell_type'] != 'code': continue

            # skip deliberately failing cells
            if BOUND_TO_FAIL in c['source']: continue

            # skip multiprocessing cells
            if any([ tabu in c['source'].split() for tabu in [
                'pathos.pools',
                'mulitprocessing',
            ]]): 
                print('\n'.join([
                    f'\nskip multiprocessing cell {i} in {self.notebook}',
                    '+'+'-'*68+'+',
                    c['source']
                ]))
                continue

            # remove non python lines and help calls which require user input
            python_code = "\n".join([ln for ln in c['source'].split("\n") 
                if not ln.startswith('%')
                and not ln.startswith('help(')
                and not ln.startswith('ask_ok(')
                and not ln.strip().endswith('?')
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
                self.fail(failure)

        print(f'succeeded {self.notebook}')


def main():
    # list notebooks in the NOTEBOOK_DIR
    notebooks = [(NOTEBOOK_DIR, f)
                 for f in sorted(os.listdir(NOTEBOOK_DIR))
                 if os.path.splitext(f)[1] == ('.ipynb')]

    # build a test suite with a test for each notebook
    suite = unittest.TestSuite()
    for (jd,nb) in notebooks:
        suite.addTest(NotebookTest('test_notebook', jd, nb))
    
    # run the tests depending on the first input argument: None or 'report'. 
    # write xml reports for 'report'
    if sys.argv[1:]:
        arg = sys.argv[1]
        if arg == 'report':
            import xmlrunner
            output = os.path.join(SOURCE_DIR, '../tests_xml')
            xmlrunner.XMLTestRunner(output=output).run(suite)
        else:
            jd, nb = os.path.split(arg)
            unittest.TextTestRunner(verbosity=2).run(NotebookTest('test_notebook', jd, nb))
    # with no argument just run the test
    else:
        unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    main()

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import unittest
import nbformat

from climada.util.constants import SOURCE_DIR
NOTEBOOK_DIR = os.path.abspath('doc/tutorial')


class NotebookTest(unittest.TestCase):
    '''Generic TestCase for testing the executability of notebooks'''

    def __init__(self, methodName, wd=None, notebook=None):
        super(NotebookTest, self).__init__(methodName)
        self.wd = wd
        self.notebook = notebook

    def test_notebook(self):
        '''Extracts code cells from the notebook and executes them one by one, using `exec`.
        Magic lines and help/? calls are eliminated.'''

        # cd to the notebook directory
        os.chdir(self.wd)
        print(f'start testing {self.notebook}')

        # read the notebook into a string
        with open(self.notebook, encoding='utf8') as nb:
            content = nb.read()
        
        # parse the string with nbformat.reads
        cells = nbformat.reads(content, 4)['cells']
        
        for i, c in enumerate(cells):
            # skip markdown cells
            if c['cell_type'] != 'code': continue

            # remove non python lines and help calls which require user input
            python_code = "\n".join([ln for ln in c['source'].split("\n") 
                if not ln.startswith('%matplotlib')
                and not ln.startswith('help(')
                and not ln.strip().endswith('?')
            ])

            # execute the python code
            try:
                exec(python_code)
            
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
    notebooks = [(NOTEBOOK_DIR, f) for f in os.listdir(NOTEBOOK_DIR) if os.path.splitext(f)[1] == ('.ipynb')]

    # build a test suite with a test for each notebook
    suite = unittest.TestSuite()
    for (jd,nb) in notebooks:
        suite.addTest(NotebookTest('test_notebook', jd, nb))
    
    # run the tests depending on the first input argument: None or 'report'. 
    # write xml reports for 'report'
    if sys.argv[1:]:
        import xmlrunner
        arg = sys.argv[1]
        if arg == 'report':
            output = os.path.join(SOURCE_DIR, '../tests_xml')
            xmlrunner.XMLTestRunner(output=output).run(suite)
    # with no argument just run the test
    else:
        unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    main()

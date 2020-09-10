#!/usr/bin/env python
# coding: utf-8

import os
import sys
import unittest
import nbformat


ROOT = os.path.abspath(os.path.dirname(__file__))
NOTEBOOK_DIR = 'doc/tutorial'


class NotebookTest(unittest.TestCase):
'''TestCase for testing the executability of notebooks'''
    def __init__(self, methodName, notebook=None):
        super(NotebookTest, self).__init__(methodName)
        self.notebook = notebook

    def test_notebook(self):
    '''Extracts code cells from the notebook and executes them one by one, using `exec`.
    Magic lines and help/? calls are eliminated.'''
        # cd to the top directory (you never know what happens in exec)
        os.chdir(ROOT)
        print(f'start testing {self.notebook}')

        # read the notebook into a string
        with open(f'{NOTEBOOK_DIR}/{self.notebook}', encoding='utf8') as nb:
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
    notebooks = [f for f in os.listdir(NOTEBOOK_DIR) if os.path.splitext(f)[1] == ('.ipynb')]

    # build a test suite with a test for each notebook
    suite = unittest.TestSuite()
    for nb in notebooks:
        suite.addTest(NotebookTest('test_notebook', nb))
    
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

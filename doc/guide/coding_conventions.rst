.. _Coding Conventions:

Coding Conventions
==================

Contributions are very welcome! But we need to keep a certain order. Please earnestly consider the following guidelines.

Unit Tests
----------
Each method/function should have its dedicated unit test suit.
Excepted are methods/functions that are only called in a particular, isolated context and not meant to be called from elsewhere.
For these cases it seems sufficient to test the calling method/function.


Python Style
------------
Follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.

Linter
------
The PyLint configuration file is `here <https://github.com/CLIMADA-project/climada_python/blob/master/.pylintrc>`_.
As a general rule, aim for no warnings at all and always make sure *High Priority* warnings are immediately eliminated.


Best Practices
==============
Coding can be good or bad, this much is clear. It's not so clear what exactly distinguishes the good from the bad.
This question has probably only one correct answer, the universal: it depends.

In the context of CLIMADA, we consider code to be good if it adheres to the following commitments at best effort, i.e. reasonably rather than dogmatic.

Correctness
-----------
Methods and functions must return correct and verifiable results, not only under the best circumstances but in any possible context.
I.e. ideally there should be unit tests exploring the full space of parameters, configuration and data states.
This is often clearly a non-achievable goal, but still - we aim at it.

Tightness
---------
- Avoid code redundancy.
- Make the program efficient, use profiling tools for detection of bottlenecks.
- Try to minimize memory consumption.
- Don't introduce new dependencies (library imports) when the desired functionality is already covered by existing dependencies.
- Stick to already supported file types.

Readability
-----------
- Write complete Python Docstrings.
- Use meaningful method and parameter names, and always annotate the data types of parameters and return values.
- No context depending return types! Avoid None as return type, rather raise an Exception instead. ???
- Be generous with defining Exception classes.
- Comment! Comments are welcome to be redundant.
  And whenever there is a particular reason for the way something is done, comment on it!
  It *will* pay off when maintaining, extending or debugging. An extensive guide is `here <https://realpython.com/python-comments-guide/#when-writing-code-for-others>`_.
- For functions which implement mathematical/scientific concepts, add the actual mathematical formula as comment or
  to the Doctstrings. This will help maintain a high level of scientific accuracy. E.g. How is are the random walk
  tracks computed for tropical cyclones?

Performance
===========
C-like data types
-----------------
- Use arrays, implicitly (DataFrames) or explicitly.
- Initialize arrays and DataFrames not by appending to an initially empty array or DataFrame but
  by using concatenation (`numpy.vstack`, `pandas.concat`) of lists or maps.
- Avoid loops (`for`, `while`) around arrays and DataFrames in favor of
  vectorized operations.
- Mind the creation of temporary arrays in vector arithmetics.

Parallelization
---------------
Don't parallelize inefficient programs! (Unless they're not yours and you cannot change them.)

Cython, Numba, ...
------------------
- First try to exploit Numpy vectorized operations to speed up your code before you resort to tools like Cython or Numba.
- When using Numba, make sure to avoid Python objects as, otherwise, Numba will
  use the less efficient `object mode <https://numba.pydata.org/numba-doc/latest/glossary.html#term-object-mode>`_.

Configuration
=============
- URLs of external resources and locations of data directories should always be defined in the config.py file and not declared as constants.

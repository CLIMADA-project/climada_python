.. _Coding Conventions:

Coding Conventions
==================

Contributions are very welcome! But we need to keep a certain order. Please seriously consider the following guide lines.

Unit Tests
----------
Each method/function should have its dedicated unit test suit.
Excepted are methods/fonctions that are only called in a particular, isolated context and not meant to be called from elsewhere.
For these cases it seems sufficient to test the calling method/function.


Python Style
------------
Follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.

Linter
------
The PyLint configuration file is `here <https://github.com/CLIMADA-project/climada_python/blob/master/.pylintrc>`_.
As a general rule, aim for no warnings at all and allways make sure *High Priority* warnings are immediately eliminated.

Branches
--------
In general policy and naming of branches follow the conventions of `git flow <https://nvie.com/posts/a-successful-git-branching-model/>`_.

The table below contains an overview of tolerance toward offences against unittest and linter.
 
======= ===== ======= ==== ====== ===
Branch  Unittest          Linter
------- ------------- ---------------
\       Error Failure High Medium Low
======= ===== ======= ==== ====== ===
Master  x     x       x    \(x\)  \-
Develop x     n days  x    m days \-
Feature \(x\) \-      \-   \-     \-
======= ===== ======= ==== ====== ===

x indicates "no tolerance", meaning that any code changes producing such offences should be fixed *before* pushing them
to the respective branch.


Best Practices
==============
Coding can be good or bad, so much is clear. Not so clear is what exactly diestinguishes the good from the bad. 
This question has probably only one correct answer, the universal: it depends.

In the context of CLIMADA, we consider code to be good if it adheres to the following commitments at best effort, i.e. reasonably rather than dogmatic.

Correctness
-----------
The methods must return correct and verifiable results, not only under the best circumstances but in any possible context. 
I.e. ideally there should be unit tests exploring the full space of parameters, configuration and data states. This is often clearly a non-achievable goal, but still - we aim at it.

Tightness
---------
- Avoid code redundancy.
- Make the program efficient, use profiling tools for detection of bottlenecks.
- Try to minimize memory consumption.
- Don't introduce new dependencies (library imports) when the desired functionality is already covered by existing dependencies.
- Stick to already supported file types.

Readability
-----------
- Write complete Python-Doc strings.
- Use meaningful method and parameter names, and always annotate the data types of parameters and return values.
- No context depending return types! Avoid None as return type, rather raise an Exception instead. ???
- Be generous with with defining Exception classes.
- Comment! Comments are welcome to be redundant.
  And whenever there is a particular reason for the way something is done, comment on it!
  It *will* pay off when maintaining, extending or debugging.


Performance
===========
C-like data types
-----------------
- Use arrays, implicitly (DataFrames) or explicitly.
- Initialize arrays and DataFrames not by incrementing the array or DataFrame but by 
  creating them from lists or maps.
- Avoid `for` loops around arrays and DataFrames.
- Mind the creation of temporary arrays in vector arithmetics.

Parallelization
---------------
Don't parallelize inefficient programs! (Unless they're not yours and you cannot change them.)

Cython, Numba, ...
------------------
???

Exposed Interface
=================
Library functions???

Configuration
=============
Dealing with constants
----------------------
??? central registry?

Dealing with defaults
---------------------
??? 
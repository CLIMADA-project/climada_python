# Reviewer Checklist

* Include references to the used algorithms in the docstring
* If the algorithm is new, please include a description in the docstring, or be sure to include a reference as soon as you publish the work
* The code should be easily readable (for infos e.g. here [here](https://treyhunner.com/2017/07/craft-your-python-like-poetry/?__s=jf8h91lx6zhl7vv6o9jo))
* Variable names should be chosen to be clear. Avoid `item, element, var, list` etc... 
* Avoid as much as possible hard-coded indices for list (no `x = l[0], y = l[1]`) (see also [here](https://treyhunner.com/2018/03/tuple-unpacking-improves-python-code-readability/))
* Avoid mutable as default values for functions and methods.
* Use pythonic loops, list comprehensions etc.
* Make sure the unit test are testing all the relevant parts of the code
* Check the docstring (is everything clearly explained, are the default values given an clear)

* Did the code writer perform a static code analysis? Does the code respect Pep8?
* Did the code writer perform a profiling and checked that there are no obviously ineficient (computation time-wise and memore-wise) parts in the code?

Reviewer Checklist
==================

-  The code must be readable without extra effort from your part. The
   code should be easily readable (for infos e.g. here
   `here <https://treyhunner.com/2017/07/craft-your-python-like-poetry/?__s=jf8h91lx6zhl7vv6o9jo>`__)
-  Include references to the used algorithms in the docstring
-  If the algorithm is new, please include a description in the
   docstring, or be sure to include a reference as soon as you publish
   the work
-  Variable names should be chosen to be clear. Avoid
   ``item, element, var, list, data`` etc... A good variable name makes
   it immediately clear what it contains.
-  Avoid as much as possible hard-coded indices for list (no
   ``x = l[0], y = l[1]``). Rather, use tuple unpacking (see
   `here <https://treyhunner.com/2018/03/tuple-unpacking-improves-python-code-readability/>`__).
   Note that tuple unpacking can also be used to update variables. For
   example, the Fibonacci sequence next number pair can be written as
   ``n1, n2 = n2, n1+n2``.
-  Do not use
   `mutable <https://www.geeksforgeeks.org/mutable-vs-immutable-objects-in-python/>`__
   (lists, dictionnaries, ...) as `default values for functions and
   methods <https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument>`__.
   Do not write: 
   ::
   	def function(default=[]):
   	    ... 
   but use
   ::
   	def function(default=None):
     	    if default is None: default=[]

-  Use pythonic loops, `list
   comprehensions <https://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/>`__
   etc.
-  Make sure the unit test are testing all the lines of the code. Do not
   only check for working cases, but also the most common wrong use
   cases.
-  Check the docstring (is everything clearly explained, are the default
   values given and is it clear why they are set to this value)
-  Keep the code simple. Avoid using complex Python functionalities whose use
   is oppaque to non-expert developers unless necessary. For example, the
   ``@staticmethod`` decorator should only be used if really 
   necessary. Another example, for counting the dictionnary
   ``colors = ['red', 'green', 'red', 'blue', 'green', 'red']``,
   version
   ::
   	d = {}
	for color in colors:
            d[color] = d.get(color, 0) + 1 
   is perfectly fine, no need to complicate it to a maybe more pythonic
   version
   ::
   	d = collections.defaultdict(int)
     	for color in colors:
            d[color] += 1

-  Did the code writer perform a static code analysis? Does the code
   respect Pep8 (see also the `pylint config file <https://github.com/CLIMADA-project/climada_python/blob/main/.pylintrc/>`__)?
-  Did the code writer perform a profiling and checked that there are no
   obviously ineficient (computation time-wise and memore-wise) parts in
   the code?



"""
=====================
Tag
=====================

Module containing the tag used for the hazard package.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Tue Nov 28 11:14:02 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

class Tag(object):
    """ Contains the definition of one tag """

    def __init__(self, file_name=None, description=None, haz_type=None):
        self.file_name = file_name
        self.description = description
        self.type = haz_type
        self._next = 'NA'
        self._prev = 'NA'

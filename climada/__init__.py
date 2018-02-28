"""
climada init
"""

from .util.config import setup_logging
setup_logging()

from .entity import *
from .hazard import *
from .engine import *
from .util import *

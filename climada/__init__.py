"""
climada init
"""

from .util.config import setup_logging
setup_logging()


from .util.config import CONFIG
from .util.config import setup_conf_user
setup_conf_user()

from .entity import *
from .hazard import *
from .engine import *
from .util import *

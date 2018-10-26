"""
climada init
"""

from .util.config import CONFIG
from .util.config import setup_conf_user
from .util.config import setup_logging
setup_conf_user()
setup_logging(CONFIG['global']['log_level'])

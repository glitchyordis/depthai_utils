import logging
import pathlib
from attrs import define, frozen

logs_dir = pathlib.Path.cwd()/ "logs"
if not logs_dir.exists():
    logs_dir.mkdir()

logger = logging.getLogger('polaris_logger') # Create a logger

# Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG) 

# a file handler that writes messages to a file
file_handler = logging.FileHandler(logs_dir/'polaris.log') 

# Set the logging level for the file handler
file_handler.setLevel(logging.DEBUG)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler.setFormatter(formatter) # formatter to the file handler
logger.addHandler(file_handler) # file handler to the logger

"""
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')
"""

@define(slots = True, frozen=True)
class LogLevelKeys():
    info : str = "info"
    exception : str = "exception"
    debug : str = "debug"
    warning : str = "warning"
    critical : str = "critical"
    
log_level_keys = LogLevelKeys()

LOG_LEVEL_MAPPING = {
    log_level_keys.info: logger.info,
    log_level_keys.exception: logger.exception,
    log_level_keys.debug: logger.debug,
    log_level_keys.warning: logger.warning,
    log_level_keys.critical: logger.critical
}
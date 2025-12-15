import logging
import os

# Assume project_root can be determined here or passed in.
# For simplicity, let's derive it relative to this utility file
_current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_current_dir, '..', '..'))

# --- Global Logging Setup ---
LOG_FILE_PATH = os.path.join(project_root, "training_debug.log")

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# This is the single, accessible logger object
logger = logging.getLogger('project_logger')

# Also set up a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)
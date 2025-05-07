import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
def setup_logging(module_name):
    # Create a logger with the module name
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Create handlers with module name in filename
    current_time = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(f'logs/{module_name}_{current_time}.log')
    
    # Create formatters with filename info
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)

    return logger

# Example usage:
# In other files, import and call like:
# from logging_config import setup_logging
# logger = setup_logging(__name__)  # __name__ will be the module's name
import logging

# Set up logger for this package
logger = logging.getLogger("ui")

def print_color(message, color):
    # Function for displaying color messages - now logs instead of printing
    if color == "red":
        logger.error(message)
    elif color == "yellow":
        logger.warning(message)
    else:
        logger.info(message)

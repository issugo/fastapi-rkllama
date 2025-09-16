import logging


class LoggerConsoleFormatter(logging.Formatter):

    black = "\033[30m"
    red = "\033[31m"
    orange = "\033[33m"
    blue = "\033[34m"
    green = "\033[92m"
    yellow = "\033[93m"
    bold = "\033[01m"
    reset = "\033[00m"

    log_format = r'%(asctime)s.%(msecs)03d %(levelname)5s %(name)s [%(filename)s:%(lineno)d] - %(message)s'
    date_format = r'%Y-%m-%dT%H:%M:%S'

    FORMATS = {
        5: blue + log_format + reset,
        logging.DEBUG: orange + log_format + reset,
        logging.INFO: black + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold + red + log_format + reset,
    }

    def format(self, record):
       log_fmt = self.FORMATS.get(record.levelno)
       formatter = logging.Formatter(log_fmt, self.date_format
       formatted = formatter.format(record)
       return formatted

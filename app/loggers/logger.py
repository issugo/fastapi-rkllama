import logging
import os

from loggers.loggerRawFormatter import LoggerRawFormatter
from loggers.loggerConsoleFormatter import LoggerConsoleFormatter

class Logger:

    # OVERRIDE LOG LEVELS TO INPUT CUSTOM LEVELS
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    TRACE = 5
    NOTSET = 0

    nameToLevel = {
        'CRITICAL': CRITICAL,
        'FATAL': FATAL,
        'ERROR': ERROR,
        'WARNING': WARNING,
        'WARN': WARN,
        'INFO': INFO,
        'DEBUG': DEBUG,
        'TRACE': TRACE,
        'NOTSET': NOTSET
    }

    @staticmethod
    def get_level():
        level = os.environ.get("LOG_LEVEL", "INFO")
        if isinstance(level, int):
            rv = level
        elif str(level) == level:
            if level not in Logger.nameToLevel:
                raise ValueError("Unknown level: %r" % level)
            rv = Logger.nameToLevel[level]
        else:
            raise TypeError("LOG_LEVEL not an integer or a valid string: %r" % (level,))
        return rv

    @staticmethod
    def get_filename():
        return os.environ.get("LOG_FILE", "logs.log")

    @staticmethod
    def addCustomLevel(levelNum, levelName, methodName=None):
        if not methodName:
            methodName = levelName.lower()

        def logForLevel(self, message, *args, **kwargs):
            if self.isEnabledFor(levelNum):
                self._log(levelNum, message, *args, **kwargs)

        def logToRoot(message, *args, **kwargs):
            logging.log(levelNum, message, *args, **kwargs)

        logging.addLevelName(levelNum, levelName)
        setattr(logging, levelName, levelNum)
        setattr(logging.getLoggerClass(), methodName, logForLevel)
        setattr(logging, methodName, logToRoot)

    @staticmethod
    def get_logger(name):
        # ADD TRACE(5) CUSTOM LOG LEVEL
        Logger.addCustomLevel(5, "TRACE")

        # GetLogger and configure it
        logger = logging.getLogger(name)
        logger.setLevel(Logger.get_level())

        file_hdlr = logging.FileHandler(Logger.get_filename())
        file_hdlr.setFormatter(LoggerRawFormatter())
        logger.addHandler(hdlr=file_hdlr)

        console_hdlr = logging.StreamHandler()
        console_hdlr.setFormatter(LoggerConsoleFormatter())
        logger.addHandler(hdlr=console_hdlr)

        return logger

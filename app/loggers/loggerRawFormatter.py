import logging


class LoggerRawFormatter(logging.Formatter):
    log_format = r"%(asctime)s.%(msecs)03d %(levelname)5s %(name)s [%(filename)s:%(lineno)d] - %(message)s"
    date_format = r"%Y-%m-%dT%H:%M:%S"

    def format(self, record):
        formatter = logging.Formatter(self.log_format, self.date_format)
        formatted = formatter.format(record)
        return formatted

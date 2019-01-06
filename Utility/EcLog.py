import logging
from  logging.handlers import  RotatingFileHandler

class eclog:
    def __init__(self, file):
        self.eclogger = logging.getLogger(file)
        self.eclogger.setLevel(level=logging.DEBUG)
        if not self.eclogger.handlers:
            self.rotate_handler = RotatingFileHandler("FECA.log", maxBytes=1024 * 1024, backupCount=5)
            self.rotate_handler.setLevel(level=logging.DEBUG)
            DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
            formatter = logging.Formatter(fmt='%(asctime)s(File:%(name)s,Line:%(lineno)d, %(funcName)s) - %(levelname)s - %(message)s', datefmt=DATE_FORMAT)

            self.rotate_handler.setFormatter(formatter)
            self.eclogger.addHandler(self.rotate_handler)

    def GetLogger(self):
        return self.eclogger;



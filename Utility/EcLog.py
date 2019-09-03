import logging
import copy
from logging.handlers import RotatingFileHandler

class eclog(object):
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

    # def __deepcopy__(self, memodict={}):
    #     copy_object = type(self)("new_FECA.log")
    #     copy_object.rotate_handler = copy.deepcopy(self.rotate_handler, memodict)
    #     copy_object.eclogger = copy.deepcopy(self.eclogger, memodict)
        # return copy_object

    def GetLogger(self):
        return self.eclogger


if __name__ == '__main__':
    logger = eclog('temp.log')
    new_logger = copy.deepcopy(logger)
    new_logger.GetLogger().error('test')
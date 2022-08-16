import logging
import copy
from logging.handlers import RotatingFileHandler
import threading

class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with Singleton._instance_lock:
                if not hasattr(cls, '_instance'):
                    Singleton._instance = super().__new__(cls)

        return Singleton._instance

class eclog(Singleton):
    def __init__(self, file):
        Singleton.__init__(self)
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

    #def LogInfo(self, info):
     #   self.eclogger.info(info)

   # def LogError(self, info):
     #   self.eclogger.error(info)

if __name__ == '__main__':
    #eclog("t").LogInfo('test')
    #eclog("t").LogError('error test')
    eclog("t").GetLogger().error('test')
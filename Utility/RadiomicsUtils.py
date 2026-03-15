import logging

import radiomics


def silence_radiomics_logging():
    radiomics.setVerbosity(logging.ERROR)

    radiomics_logger = logging.getLogger('radiomics')
    radiomics_logger.setLevel(logging.ERROR)
    radiomics_logger.propagate = False

    has_null_handler = any(isinstance(handler, logging.NullHandler) for handler in radiomics_logger.handlers)
    if not has_null_handler:
        radiomics_logger.addHandler(logging.NullHandler())

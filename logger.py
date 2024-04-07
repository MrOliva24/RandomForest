import logging


# Logger used to follow the process that is running everytime, and be able to see where occurs any problem if needed


def mylogger(name: str, level: int):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    # output goes to console
    formatter = logging.Formatter('%(name)s.py %(lineno)d - %(asctime)s'
                                  ' - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # otherwise root's handler is used => as if level was WARNING
    return logger

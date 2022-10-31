import logging

from BertMatch.config import BaseOptions


def get_logger(name='root'):
    logger = logging.getLogger(name)
    # config = BaseOptions().parse()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y%m%d %H:%M:%S %p")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # file_handler = logging.FileHandler(config.train_log_filepath, mode='w')
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    return logger

logger = get_logger(__name__)
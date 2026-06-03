import os
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("files")


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def join(*args):
    return os.path.join(*args).replace("\\", "/")


def dirname(path, level=1):
    for i in range(level):
        path = os.path.dirname(path)

    return path


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

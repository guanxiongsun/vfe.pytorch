# Copyright (c) OpenMMLab. All rights reserved.
import logging
import timeit
from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger


class SpeedTester:
    def __init__(self, log_file, log_level):
        self.logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)
        self.runtime_dict = {}
        self.start_time_dict = {}
        self.end_time_dict = {}

    def reset(self):
        self.runtime_dict = {}
        self.start_time_dict = {}
        self.end_time_dict = {}

    def start_of_key(self, key):
        if key not in self.runtime_dict.keys():
            # register
            self.runtime_dict.update({key: []})

        self.start_time_dict.update({
            key: timeit.default_timer()
        })

    def end_of_key(self, key, record=True):
        self.end_time_dict.update({
            key: timeit.default_timer()
        })
        # record runtime of key
        if record:
            self.runtime_dict[key].append(
                self.end_time_dict[key] - self.start_time_dict[key]
            )


speed_tester = None


def get_speed_tester(log_file=None, log_level=logging.INFO):
    global speed_tester
    if log_file is None:
        assert isinstance(speed_tester, SpeedTester)
    else:
        speed_tester = SpeedTester(log_file, log_level)

    return speed_tester

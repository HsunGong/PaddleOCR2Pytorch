import os
import sys
import logging
import functools
import torch.distributed as dist

import logging
from typing import Any, List, Optional

logger_initialized = {}


@functools.lru_cache()
def get_logger(name='root', log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # if dist.get_rank() == 0:
    logger.setLevel(log_level)
    # else:
    #     logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    return logger

logger = get_logger("pytorchocr", log_level=logging.INFO)

def redirect_to_logger(*args: Any, sep: Optional[str] = ' ', end: Optional[str] = '\n', level: int = logging.INFO) -> None:
    """
    功能与print类似，但会把输出重定向到logging模块。
    能够处理print支持的所有参数，并转化为一条logging记录。

    参数:
        *args: 要打印的对象。
        sep: 分隔符，用于分隔多个对象，默认是空格。
        end: 结尾字符串，添加到最后一个对象之后，默认是换行符。
        logger: 要使用的logger实例，若未提供则使用root logger。
        level: 日志级别，默认是INFO。
    """
    # logger = logging.getLogger("pytorchocr")
    str_args: List[str] = [str(arg) for arg in args]
    combined_str: str = sep.join(str_args) + end
    logger.log(level, combined_str.rstrip('\n'))
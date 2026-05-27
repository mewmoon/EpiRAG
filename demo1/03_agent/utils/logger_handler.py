import logging
import sys
import os
import datetime
from path_utils import PathUtils

LOG_ROOT = PathUtils.to_absolute("logs")
os.makedirs(LOG_ROOT, exist_ok=True)
LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s -%(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
    name: str = "agent",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file=None,
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 防止重复添加 Handler
    if logger.handlers:
        return logger

    # 控制台Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(LOG_FORMAT)
    console_handler.setLevel(console_level)

    logger.addHandler(console_handler)

    # 文件Handler
    if not log_file:
        log_file = os.path.join(
            LOG_ROOT, f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(LOG_FORMAT)
    file_handler.setLevel(file_level)

    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    print(LOG_ROOT)
    logger = get_logger()
    logger.info("info信息日志")
    logger.debug("debug调试日志")
    logger.warning("warning警告日志")
    logger.error("error错误日志")

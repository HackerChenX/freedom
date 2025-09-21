"""
Global setting of the trading platform.
"""

from logging import CRITICAL
from tzlocal import get_localzone_name

from .utility import load_json


SETTINGS: dict = {
    "font.family": "微软雅黑",
    "font.size": 12,

    "log.active": True,
    "log.level": CRITICAL,
    "log.console": True,
    "log.file": True,

    "email.server": "smtp.qq.com",
    "email.port": 465,
    "email.username": "",
    "email.password": "",
    "email.sender": "",
    "email.receiver": "",

    "datafeed.name": "efinance",
    "datafeed.username": "",
    "datafeed.password": "",

    "database.timezone": "Asia/Shanghai",
    "database.name": "clickhouse",
    "database.database": "stock",
    "database.host": "localhost",
    "database.port": 8123,
    "database.user": "default",
    "database.password": "123456"
}


# Load global setting from json file.
SETTING_FILENAME: str = "vt_setting.json"
# SETTINGS.update(load_json(SETTING_FILENAME))

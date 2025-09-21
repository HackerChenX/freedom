"""
数据查看器应用主模块
遵循VnPy应用架构设计
"""

from pathlib import Path
from vnpy.trader.app import BaseApp
from vnpy.trader.constant import Direction
from vnpy.trader.object import TickData, BarData
from vnpy.trader.utility import load_json, save_json

from .engine import DataViewerEngine


APP_NAME = "DataViewer"


class DataViewerApp(BaseApp):
    """数据查看器应用"""

    app_name: str = APP_NAME
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "数据查看器"
    engine_class: type[DataViewerEngine] = DataViewerEngine
    widget_name: str = "DataViewerWidget"
    icon_name: str = str(app_path.joinpath("data_viewer.ico"))

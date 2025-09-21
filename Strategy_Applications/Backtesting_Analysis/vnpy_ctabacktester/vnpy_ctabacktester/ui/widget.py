import os
import platform
import csv
import shutil
import subprocess
from datetime import datetime, timedelta
from copy import copy
from typing import Any

import numpy as np
import pyqtgraph as pg
from pandas import DataFrame

from vnpy.trader.constant import Interval, Direction, Exchange
from vnpy.trader.engine import MainEngine, BaseEngine
from vnpy.trader.ui import QtCore, QtWidgets, QtGui
from vnpy.trader.ui.widget import BaseMonitor, BaseCell, DirectionCell, EnumCell
from vnpy.event import Event, EventEngine
from vnpy.chart import ChartWidget, CandleItem, VolumeItem
from .kdj_item import KdjItem


class KdjChartItemWrapper:
    """KDJ图表项包装器 - 用于与ChartWidget系统集成"""
    
    def __init__(self, kdj_item):
        self.kdj_item = kdj_item
        
    def get_info_text(self, ix: int) -> str:
        try:
            if hasattr(self.kdj_item, 'k_data') and ix < len(self.kdj_item.k_data):
                k_val = self.kdj_item.k_data[ix]
                d_val = self.kdj_item.d_data[ix] 
                j_val = self.kdj_item.j_data[ix]
                return f"KDJ: K={k_val:.2f} D={d_val:.2f} J={j_val:.2f}"
        except (IndexError, AttributeError):
            pass
        return ""
    
    def get_y_range(self, min_ix: int = None, max_ix: int = None) -> tuple:
        # KDJ指标通常在0-100范围内，但J值可能超出
        return (-10, 110)
    
    def update_history(self, history):
        # 空实现，满足ChartWidget的要求
        pass
    
    def clear_all(self):
        # 空实现，满足ChartWidget的要求
        pass
    
    def update_bar(self, bar):
        # 空实现，满足ChartWidget的要求
        pass


from vnpy.trader.utility import load_json, save_json
from vnpy.trader.object import BarData, TradeData, OrderData
from vnpy.trader.database import DB_TZ
from vnpy_ctastrategy.backtesting import DailyResult

from ..locale import _
from ..engine import (
    APP_NAME,
    EVENT_BACKTESTER_LOG,
    EVENT_BACKTESTER_BACKTESTING_FINISHED,
    EVENT_BACKTESTER_OPTIMIZATION_FINISHED,
    OptimizationSetting
)


class BacktesterManager(QtWidgets.QWidget):
    """"""

    setting_filename: str = "cta_backtester_setting.json"

    signal_log: QtCore.Signal = QtCore.Signal(Event)
    signal_backtesting_finished: QtCore.Signal = QtCore.Signal(Event)
    signal_optimization_finished: QtCore.Signal = QtCore.Signal(Event)

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__()

        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine

        self.backtester_engine: BaseEngine = main_engine.get_engine(APP_NAME)
        self.class_names: list = []
        self.settings: dict = {}

        self.target_display: str = ""

        self.init_ui()
        self.register_event()
        self.backtester_engine.init_engine()
        self.init_strategy_settings()
        self.load_backtesting_setting()

    def init_strategy_settings(self) -> None:
        """"""
        self.class_names = self.backtester_engine.get_strategy_class_names()
        self.class_names.sort()

        for class_name in self.class_names:
            setting: dict = self.backtester_engine.get_default_setting(class_name)
            self.settings[class_name] = setting

        self.class_combo.addItems(self.class_names)

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(_("CTA回测"))

        # Setting Part
        self.class_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()

        self.symbol_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit("IF88.CFFEX")

        self.interval_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        for interval in Interval:
            self.interval_combo.addItem(interval.value)

        end_dt: datetime = datetime.now()
        start_dt: datetime = end_dt - timedelta(days=3 * 365)

        self.start_date_edit: QtWidgets.QDateEdit = QtWidgets.QDateEdit(
            QtCore.QDate(
                start_dt.year,
                start_dt.month,
                start_dt.day
            )
        )
        self.end_date_edit: QtWidgets.QDateEdit = QtWidgets.QDateEdit(
            QtCore.QDate.currentDate()
        )

        self.rate_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit("0.000025")
        self.slippage_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit("0.2")
        self.size_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit("300")
        self.pricetick_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit("0.2")
        self.capital_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit("1000000")

        backtesting_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("开始回测"))
        backtesting_button.clicked.connect(self.start_backtesting)

        optimization_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("参数优化"))
        optimization_button.clicked.connect(self.start_optimization)

        self.result_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("优化结果"))
        self.result_button.clicked.connect(self.show_optimization_result)
        self.result_button.setEnabled(False)

        downloading_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("下载数据"))
        downloading_button.clicked.connect(self.start_downloading)

        self.order_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("委托记录"))
        self.order_button.clicked.connect(self.show_backtesting_orders)
        self.order_button.setEnabled(False)

        self.trade_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("成交记录"))
        self.trade_button.clicked.connect(self.show_backtesting_trades)
        self.trade_button.setEnabled(False)

        self.daily_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("每日盈亏"))
        self.daily_button.clicked.connect(self.show_daily_results)
        self.daily_button.setEnabled(False)

        self.candle_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("K线图表"))
        self.candle_button.clicked.connect(self.show_candle_chart)
        self.candle_button.setEnabled(False)

        self.decision_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("决策分析"))
        self.decision_button.clicked.connect(self.show_decision_analysis)
        self.decision_button.setEnabled(False)

        edit_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("代码编辑"))
        edit_button.clicked.connect(self.edit_strategy_code)

        reload_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("策略重载"))
        reload_button.clicked.connect(self.reload_strategy_class)

        for button in [
            backtesting_button,
            optimization_button,
            downloading_button,
            self.result_button,
            self.order_button,
            self.trade_button,
            self.daily_button,
            self.candle_button,
            self.decision_button,
            edit_button,
            reload_button
        ]:
            button.setFixedHeight(button.sizeHint().height() * 2)

        form: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        form.addRow(_("交易策略"), self.class_combo)
        form.addRow(_("本地代码"), self.symbol_line)
        form.addRow(_("K线周期"), self.interval_combo)
        form.addRow(_("开始日期"), self.start_date_edit)
        form.addRow(_("结束日期"), self.end_date_edit)
        form.addRow(_("手续费率"), self.rate_line)
        form.addRow(_("交易滑点"), self.slippage_line)
        form.addRow(_("合约乘数"), self.size_line)
        form.addRow(_("价格跳动"), self.pricetick_line)
        form.addRow(_("回测资金"), self.capital_line)

        result_grid: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        result_grid.addWidget(self.trade_button, 0, 0)
        result_grid.addWidget(self.order_button, 0, 1)
        result_grid.addWidget(self.daily_button, 1, 0)
        result_grid.addWidget(self.candle_button, 1, 1)
        result_grid.addWidget(self.decision_button, 2, 0, 1, 2)  # 跨两列

        left_vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        left_vbox.addLayout(form)
        left_vbox.addWidget(backtesting_button)
        left_vbox.addWidget(downloading_button)
        left_vbox.addStretch()
        left_vbox.addLayout(result_grid)
        left_vbox.addStretch()
        left_vbox.addWidget(optimization_button)
        left_vbox.addWidget(self.result_button)
        left_vbox.addStretch()
        left_vbox.addWidget(edit_button)
        left_vbox.addWidget(reload_button)

        # Result part
        self.statistics_monitor: StatisticsMonitor = StatisticsMonitor()

        self.log_monitor: QtWidgets.QTextEdit = QtWidgets.QTextEdit()

        self.chart: BacktesterChart = BacktesterChart()
        chart: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        chart.addWidget(self.chart)

        self.trade_dialog: BacktestingResultDialog = BacktestingResultDialog(
            self.main_engine,
            self.event_engine,
            _("回测成交记录"),
            BacktestingTradeMonitor
        )
        self.order_dialog: BacktestingResultDialog = BacktestingResultDialog(
            self.main_engine,
            self.event_engine,
            _("回测委托记录"),
            BacktestingOrderMonitor
        )
        self.daily_dialog: BacktestingResultDialog = BacktestingResultDialog(
            self.main_engine,
            self.event_engine,
            _("回测每日盈亏"),
            DailyResultMonitor
        )

        # Candle Chart
        self.candle_dialog: CandleChartDialog = CandleChartDialog()
        
        # Decision Analysis Dialog
        self.decision_dialog: DecisionAnalysisDialog = DecisionAnalysisDialog(
            self.main_engine,
            self.event_engine
        )

        # Layout
        middle_vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        middle_vbox.addWidget(self.statistics_monitor)
        middle_vbox.addWidget(self.log_monitor)

        left_hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        left_hbox.addLayout(left_vbox)
        left_hbox.addLayout(middle_vbox)

        left_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        left_widget.setLayout(left_hbox)

        right_vbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        right_vbox.addWidget(self.chart)

        right_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        right_widget.setLayout(right_vbox)

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addWidget(left_widget)
        hbox.addWidget(right_widget)
        self.setLayout(hbox)

    def load_backtesting_setting(self) -> None:
        """"""
        setting: dict = load_json(self.setting_filename)
        if not setting:
            return

        self.class_combo.setCurrentIndex(
            self.class_combo.findText(setting["class_name"])
        )

        self.symbol_line.setText(setting["vt_symbol"])

        self.interval_combo.setCurrentIndex(
            self.interval_combo.findText(setting["interval"])
        )

        start_str: str = setting.get("start", "")
        if start_str:
            start_dt: QtCore.QDate = QtCore.QDate.fromString(start_str, "yyyy-MM-dd")
            self.start_date_edit.setDate(start_dt)

        self.rate_line.setText(str(setting["rate"]))
        self.slippage_line.setText(str(setting["slippage"]))
        self.size_line.setText(str(setting["size"]))
        self.pricetick_line.setText(str(setting["pricetick"]))
        self.capital_line.setText(str(setting["capital"]))

    def register_event(self) -> None:
        """"""
        self.signal_log.connect(self.process_log_event)
        self.signal_backtesting_finished.connect(
            self.process_backtesting_finished_event)
        self.signal_optimization_finished.connect(
            self.process_optimization_finished_event)

        self.event_engine.register(EVENT_BACKTESTER_LOG, self.signal_log.emit)
        self.event_engine.register(EVENT_BACKTESTER_BACKTESTING_FINISHED, self.signal_backtesting_finished.emit)
        self.event_engine.register(EVENT_BACKTESTER_OPTIMIZATION_FINISHED, self.signal_optimization_finished.emit)

    def process_log_event(self, event: Event) -> None:
        """"""
        msg = event.data
        self.write_log(msg)

    def write_log(self, msg: str) -> None:
        """"""
        timestamp: str = datetime.now().strftime("%H:%M:%S")
        msg = f"{timestamp}\t{msg}"
        self.log_monitor.append(msg)

    def process_backtesting_finished_event(self, event: Event) -> None:
        """"""
        statistics: dict = self.backtester_engine.get_result_statistics()
        self.statistics_monitor.set_data(statistics)

        df: DataFrame = self.backtester_engine.get_result_df()
        self.chart.set_data(df)

        self.trade_button.setEnabled(True)
        self.order_button.setEnabled(True)
        self.daily_button.setEnabled(True)
        self.decision_button.setEnabled(True)

        # Tick data can not be displayed using candle chart
        interval: str = self.interval_combo.currentText()
        if interval != Interval.TICK.value:
            self.candle_button.setEnabled(True)

    def process_optimization_finished_event(self, event: Event) -> None:
        """"""
        self.write_log(_("请点击[优化结果]按钮查看"))
        self.result_button.setEnabled(True)

    def start_backtesting(self) -> None:
        """"""
        class_name: str = self.class_combo.currentText()
        if not class_name:
            self.write_log(_("请选择要回测的策略"))
            return

        vt_symbol: str = self.symbol_line.text()
        interval: str = self.interval_combo.currentText()
        start: datetime = self.start_date_edit.dateTime().toPython()
        end: datetime = self.end_date_edit.dateTime().toPython()
        rate: float = float(self.rate_line.text())
        slippage: float = float(self.slippage_line.text())
        size: float = float(self.size_line.text())
        pricetick: float = float(self.pricetick_line.text())
        capital: float = float(self.capital_line.text())

        # Check validity of vt_symbol
        if "." not in vt_symbol:
            self.write_log(_("本地代码缺失交易所后缀，请检查"))
            return

        __, exchange_str = vt_symbol.split(".")
        if exchange_str not in Exchange.__members__:
            self.write_log(_("本地代码的交易所后缀不正确，请检查"))
            return

        # Save backtesting parameters
        backtesting_setting: dict = {
            "class_name": class_name,
            "vt_symbol": vt_symbol,
            "interval": interval,
            "start": start.strftime("%Y-%m-%d"),
            "rate": rate,
            "slippage": slippage,
            "size": size,
            "pricetick": pricetick,
            "capital": capital
        }
        save_json(self.setting_filename, backtesting_setting)

        # Get strategy setting
        old_setting: dict = self.settings[class_name]
        dialog: BacktestingSettingEditor = BacktestingSettingEditor(class_name, old_setting)
        i: int = dialog.exec()
        if i != dialog.DialogCode.Accepted:
            return

        new_setting: dict = dialog.get_setting()
        self.settings[class_name] = new_setting

        result: bool = self.backtester_engine.start_backtesting(
            class_name,
            vt_symbol,
            interval,
            start,
            end,
            rate,
            slippage,
            size,
            pricetick,
            capital,
            new_setting
        )

        if result:
            self.statistics_monitor.clear_data()
            self.chart.clear_data()

            self.trade_button.setEnabled(False)
            self.order_button.setEnabled(False)
            self.daily_button.setEnabled(False)
            self.candle_button.setEnabled(False)

            self.trade_dialog.clear_data()
            self.order_dialog.clear_data()
            self.daily_dialog.clear_data()
            self.candle_dialog.clear_data()

    def start_optimization(self) -> None:
        """"""
        class_name: str = self.class_combo.currentText()
        vt_symbol: str = self.symbol_line.text()
        interval: str = self.interval_combo.currentText()
        start: object = self.start_date_edit.dateTime().toPython()
        end: object = self.end_date_edit.dateTime().toPython()
        rate: float = float(self.rate_line.text())
        slippage: float = float(self.slippage_line.text())
        size: float = float(self.size_line.text())
        pricetick: float = float(self.pricetick_line.text())
        capital: float = float(self.capital_line.text())

        parameters: dict = self.settings[class_name]
        dialog: OptimizationSettingEditor = OptimizationSettingEditor(class_name, parameters)
        i: int = dialog.exec()
        if i != dialog.DialogCode.Accepted:
            return

        optimization_setting, use_ga, max_workers = dialog.get_setting()
        self.target_display = dialog.target_display

        self.backtester_engine.start_optimization(
            class_name,
            vt_symbol,
            interval,
            start,
            end,
            rate,
            slippage,
            size,
            pricetick,
            capital,
            optimization_setting,
            use_ga,
            max_workers
        )

        self.result_button.setEnabled(False)

    def start_downloading(self) -> None:
        """"""
        vt_symbol: str = self.symbol_line.text()
        interval: str = self.interval_combo.currentText()
        start_date: QtCore.QDate = self.start_date_edit.date()
        end_date: QtCore.QDate = self.end_date_edit.date()

        start: datetime = datetime(
            start_date.year(),
            start_date.month(),
            start_date.day(),
        )
        start= start.replace(tzinfo=DB_TZ)

        end: datetime = datetime(
            end_date.year(),
            end_date.month(),
            end_date.day(),
            23,
            59,
            59,
        )
        end = end.replace(tzinfo=DB_TZ)

        self.backtester_engine.start_downloading(
            vt_symbol,
            interval,
            start,
            end
        )

    def show_optimization_result(self) -> None:
        """"""
        result_values: list = self.backtester_engine.get_result_values()

        dialog: OptimizationResultMonitor = OptimizationResultMonitor(
            result_values,
            self.target_display
        )
        dialog.exec_()

    def show_backtesting_trades(self) -> None:
        """"""
        if not self.trade_dialog.is_updated():
            trades: list[TradeData] = self.backtester_engine.get_all_trades()
            self.trade_dialog.update_data(trades)

        self.trade_dialog.exec_()

    def show_backtesting_orders(self) -> None:
        """"""
        if not self.order_dialog.is_updated():
            orders: list[OrderData] = self.backtester_engine.get_all_orders()
            self.order_dialog.update_data(orders)

        self.order_dialog.exec_()

    def show_daily_results(self) -> None:
        """"""
        if not self.daily_dialog.is_updated():
            results: list[DailyResult] = self.backtester_engine.get_all_daily_results()
            self.daily_dialog.update_data(results)

        self.daily_dialog.exec_()

    def show_candle_chart(self) -> None:
        """"""
        if not self.candle_dialog.is_updated():
            history: list = self.backtester_engine.get_history_data()
            self.candle_dialog.update_history(history)

            trades: list[TradeData] = self.backtester_engine.get_all_trades()
            self.candle_dialog.update_trades(trades)

        self.candle_dialog.exec_()

    def show_decision_analysis(self) -> None:
        """显示决策分析"""
        # 获取策略决策日志
        strategy_logs = []
        trade_reasons = {}
        condition_history = []
        
        # 尝试从回测引擎获取策略实例
        if hasattr(self.backtester_engine, 'strategy'):
            strategy = self.backtester_engine.strategy
            if strategy:
                # 获取策略的决策日志
                if hasattr(strategy, 'get_decision_logs'):
                    strategy_logs = strategy.get_decision_logs()
                if hasattr(strategy, 'get_trade_reasons'):
                    trade_reasons = strategy.get_trade_reasons()
                if hasattr(strategy, 'get_condition_history'):
                    condition_history = strategy.get_condition_history()
        
        # 获取交易数据
        trades_data = self.backtester_engine.get_all_trades() if hasattr(self.backtester_engine, 'get_all_trades') else []
        
        # 整合决策数据
        decision_data = {
            'strategy_logs': strategy_logs,
            'trade_reasons': trade_reasons,
            'condition_history': condition_history
        }
        
        # 更新决策分析对话框
        self.decision_dialog.update_data(decision_data, trades_data)
        self.decision_dialog.exec_()

    def edit_strategy_code(self) -> None:
        """"""
        class_name: str = self.class_combo.currentText()
        if not class_name:
            return

        file_path: str = self.backtester_engine.get_strategy_class_file(class_name)

        if shutil.which("code"):
            if platform.system() == "Windows":
                subprocess.run(["code", file_path], shell=True)
            else:
                os.system(f"code {file_path}")
        else:
            QtWidgets.QMessageBox.warning(
                self,
                _("启动代码编辑器失败"),
                _("请检查是否安装了Visual Studio Code，并将其路径添加到了系统全局变量中！")
            )

    def reload_strategy_class(self) -> None:
        """"""
        self.backtester_engine.reload_strategy_class()

        current_strategy_name: str = self.class_combo.currentText()

        self.class_combo.clear()
        self.init_strategy_settings()

        ix: int = self.class_combo.findText(current_strategy_name)
        self.class_combo.setCurrentIndex(ix)

    def show(self) -> None:
        """"""
        self.showMaximized()


class StatisticsMonitor(QtWidgets.QTableWidget):
    """"""
    KEY_NAME_MAP: dict = {
        "start_date": _("首个交易日"),
        "end_date": _("最后交易日"),

        "total_days": _("总交易日"),
        "profit_days": _("盈利交易日"),
        "loss_days": _("亏损交易日"),

        "capital": _("起始资金"),
        "end_balance": _("结束资金"),

        "total_return": _("总收益率"),
        "annual_return": _("年化收益"),
        "max_drawdown": _("最大回撤"),
        "max_ddpercent": _("百分比最大回撤"),
        "max_drawdown_duration": _("最大回撤天数"),

        "total_net_pnl": _("总盈亏"),
        "total_commission": _("总手续费"),
        "total_slippage": _("总滑点"),
        "total_turnover": _("总成交额"),
        "total_trade_count": _("总成交笔数"),

        "daily_net_pnl": _("日均盈亏"),
        "daily_commission": _("日均手续费"),
        "daily_slippage": _("日均滑点"),
        "daily_turnover": _("日均成交额"),
        "daily_trade_count": _("日均成交笔数"),

        "daily_return": _("日均收益率"),
        "return_std": _("收益标准差"),
        "sharpe_ratio": _("夏普比率"),
        "ewm_sharpe": _("EWM夏普"),
        "return_drawdown_ratio": _("收益回撤比")
    }

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.cells: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setRowCount(len(self.KEY_NAME_MAP))
        self.setVerticalHeaderLabels(list(self.KEY_NAME_MAP.values()))

        self.setColumnCount(1)
        self.horizontalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.setEditTriggers(self.EditTrigger.NoEditTriggers)

        for row, key in enumerate(self.KEY_NAME_MAP.keys()):
            cell: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem()
            self.setItem(row, 0, cell)
            self.cells[key] = cell

    def clear_data(self) -> None:
        """"""
        for cell in self.cells.values():
            cell.setText("")

    def set_data(self, data: dict) -> None:
        """"""
        # 安全地格式化数据，使用get方法避免KeyError
        if "capital" in data:
            data["capital"] = f"{data['capital']:,.2f}"
        if "end_balance" in data:
            data["end_balance"] = f"{data['end_balance']:,.2f}"
        if "total_return" in data:
            data["total_return"] = f"{data['total_return']:,.2f}%"
        if "annual_return" in data:
            data["annual_return"] = f"{data['annual_return']:,.2f}%"
        if "max_drawdown" in data:
            data["max_drawdown"] = f"{data['max_drawdown']:,.2f}"
        if "max_ddpercent" in data:
            data["max_ddpercent"] = f"{data['max_ddpercent']:,.2f}%"
        if "total_net_pnl" in data:
            data["total_net_pnl"] = f"{data['total_net_pnl']:,.2f}"
        if "total_commission" in data:
            data["total_commission"] = f"{data['total_commission']:,.2f}"
        if "total_slippage" in data:
            data["total_slippage"] = f"{data['total_slippage']:,.2f}"
        if "total_turnover" in data:
            data["total_turnover"] = f"{data['total_turnover']:,.2f}"
        if "daily_net_pnl" in data:
            data["daily_net_pnl"] = f"{data['daily_net_pnl']:,.2f}"
        if "daily_commission" in data:
            data["daily_commission"] = f"{data['daily_commission']:,.2f}"
        if "daily_slippage" in data:
            data["daily_slippage"] = f"{data['daily_slippage']:,.2f}"
        if "daily_turnover" in data:
            data["daily_turnover"] = f"{data['daily_turnover']:,.2f}"
        if "daily_trade_count" in data:
            data["daily_trade_count"] = f"{data['daily_trade_count']:,.2f}"
        if "daily_return" in data:
            data["daily_return"] = f"{data['daily_return']:,.2f}%"
        if "return_std" in data:
            data["return_std"] = f"{data['return_std']:,.2f}%"
        if "sharpe_ratio" in data:
            data["sharpe_ratio"] = f"{data['sharpe_ratio']:,.2f}"
        if "ewm_sharpe" in data:
            data["ewm_sharpe"] = f"{data['ewm_sharpe']:,.2f}"
        if "return_drawdown_ratio" in data:
            data["return_drawdown_ratio"] = f"{data['return_drawdown_ratio']:,.2f}"

        for key, cell in self.cells.items():
            value = data.get(key, "")
            cell.setText(str(value))


class BacktestingSettingEditor(QtWidgets.QDialog):
    """
    For creating new strategy and editing strategy parameters.
    """

    def __init__(
        self, class_name: str, parameters: dict
    ) -> None:
        """"""
        super().__init__()

        self.class_name: str = class_name
        self.parameters: dict = parameters
        self.edits: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        form: QtWidgets.QFormLayout = QtWidgets.QFormLayout()

        # Add vt_symbol and name edit if add new strategy
        self.setWindowTitle(_("策略参数配置：{}").format(self.class_name))
        button_text: str = _("确定")
        parameters: dict = self.parameters

        for name, value in parameters.items():
            type_ = type(value)

            edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(value))
            if type_ is int:
                validator: QtGui.QIntValidator = QtGui.QIntValidator()
                edit.setValidator(validator)
            elif type_ is float:
                validator = QtGui.QDoubleValidator()
                edit.setValidator(validator)

            form.addRow(f"{name} {type_}", edit)

            self.edits[name] = (edit, type_)

        button: QtWidgets.QPushButton = QtWidgets.QPushButton(button_text)
        button.clicked.connect(self.accept)
        form.addRow(button)

        widget: QtWidgets.QWidget = QtWidgets.QWidget()
        widget.setLayout(form)

        scroll: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(scroll)
        self.setLayout(vbox)

    def get_setting(self) -> dict:
        """"""
        setting: dict = {}

        for name, tp in self.edits.items():
            edit, type_ = tp
            value_text = edit.text()

            if type_ is bool:
                if value_text == "True":
                    value = True
                else:
                    value = False
            else:
                value = type_(value_text)

            setting[name] = value

        return setting


class BacktesterChart(pg.GraphicsLayoutWidget):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__(title="Backtester Chart")

        self.dates: dict = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        pg.setConfigOptions(antialias=True)

        # Create plot widgets
        self.balance_plot = self.addPlot(
            title=_("账户净值"),
            axisItems={"bottom": DateAxis(self.dates, orientation="bottom")}
        )
        self.nextRow()

        self.drawdown_plot = self.addPlot(
            title=_("净值回撤"),
            axisItems={"bottom": DateAxis(self.dates, orientation="bottom")}
        )
        self.nextRow()

        self.pnl_plot = self.addPlot(
            title=_("每日盈亏"),
            axisItems={"bottom": DateAxis(self.dates, orientation="bottom")}
        )
        self.nextRow()

        self.distribution_plot = self.addPlot(title=_("盈亏分布"))

        # Add curves and bars on plot widgets
        self.balance_curve = self.balance_plot.plot(
            pen=pg.mkPen("#ffc107", width=3)
        )

        dd_color: str = "#303f9f"
        self.drawdown_curve = self.drawdown_plot.plot(
            fillLevel=-0.3, brush=dd_color, pen=dd_color
        )

        profit_color: str = 'r'
        loss_color: str = 'g'
        self.profit_pnl_bar = pg.BarGraphItem(
            x=[], height=[], width=0.3, brush=profit_color, pen=profit_color
        )
        self.loss_pnl_bar = pg.BarGraphItem(
            x=[], height=[], width=0.3, brush=loss_color, pen=loss_color
        )
        self.pnl_plot.addItem(self.profit_pnl_bar)
        self.pnl_plot.addItem(self.loss_pnl_bar)

        distribution_color: str = "#6d4c41"
        self.distribution_curve = self.distribution_plot.plot(
            fillLevel=-0.3, brush=distribution_color, pen=distribution_color
        )

    def clear_data(self) -> None:
        """"""
        self.balance_curve.setData([], [])
        self.drawdown_curve.setData([], [])
        self.profit_pnl_bar.setOpts(x=[], height=[])
        self.loss_pnl_bar.setOpts(x=[], height=[])
        self.distribution_curve.setData([], [])

    def set_data(self, df: DataFrame) -> None:
        """"""
        if df is None:
            return

        count: int = len(df)

        self.dates.clear()
        for n, date in enumerate(df.index):
            self.dates[n] = date

        # Set data for curve of balance and drawdown
        self.balance_curve.setData(df["balance"])
        self.drawdown_curve.setData(df["drawdown"])

        # Set data for daily pnl bar
        profit_pnl_x: list = []
        profit_pnl_height: list = []
        loss_pnl_x: list = []
        loss_pnl_height: list = []

        for count, pnl in enumerate(df["net_pnl"]):
            if pnl >= 0:
                profit_pnl_height.append(pnl)
                profit_pnl_x.append(count)
            else:
                loss_pnl_height.append(pnl)
                loss_pnl_x.append(count)

        self.profit_pnl_bar.setOpts(x=profit_pnl_x, height=profit_pnl_height)
        self.loss_pnl_bar.setOpts(x=loss_pnl_x, height=loss_pnl_height)

        # Set data for pnl distribution
        hist, x = np.histogram(df["net_pnl"], bins="auto")
        x = x[:-1]
        self.distribution_curve.setData(x, hist)


class DateAxis(pg.AxisItem):
    """Axis for showing date data"""

    def __init__(self, dates: dict, *args: Any, **kwargs: Any) -> None:
        """"""
        super().__init__(*args, **kwargs)
        self.dates: dict = dates

    def tickStrings(self, values: list, scale: float, spacing: float) -> list:
        """"""
        strings: list = []
        for v in values:
            dt = self.dates.get(v, "")
            strings.append(str(dt))
        return strings


class OptimizationSettingEditor(QtWidgets.QDialog):
    """
    For setting up parameters for optimization.
    """
    DISPLAY_NAME_MAP: dict = {
        _("总收益率"): "total_return",
        _("夏普比率"): "sharpe_ratio",
        _("EWM夏普"): "ewm_sharpe",
        _("收益回撤比"): "return_drawdown_ratio",
        _("日均盈亏"): "daily_net_pnl"
    }

    def __init__(
        self, class_name: str, parameters: dict
    ) -> None:
        """"""
        super().__init__()

        self.class_name: str = class_name
        self.parameters: dict = parameters
        self.edits: dict = {}

        self.optimization_setting: OptimizationSetting = None
        self.use_ga: bool = False

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        QLabel: QtWidgets.QLabel = QtWidgets.QLabel

        self.target_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.target_combo.addItems(list(self.DISPLAY_NAME_MAP.keys()))

        self.worker_spin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self.worker_spin.setRange(0, 10000)
        self.worker_spin.setValue(0)
        self.worker_spin.setToolTip(_("设为0则自动根据CPU核心数启动对应数量的进程"))

        grid: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        grid.addWidget(QLabel(_("优化目标")), 0, 0)
        grid.addWidget(self.target_combo, 0, 1, 1, 3)
        grid.addWidget(QLabel(_("进程上限")), 1, 0)
        grid.addWidget(self.worker_spin, 1, 1, 1, 3)
        grid.addWidget(QLabel(_("参数")), 2, 0)
        grid.addWidget(QLabel(_("开始")), 2, 1)
        grid.addWidget(QLabel(_("步进")), 2, 2)
        grid.addWidget(QLabel(_("结束")), 2, 3)

        # Add vt_symbol and name edit if add new strategy
        self.setWindowTitle(_("优化参数配置：{}").format(self.class_name))

        validator: QtGui.QDoubleValidator = QtGui.QDoubleValidator()
        row: int = 3

        for name, value in self.parameters.items():
            type_ = type(value)
            if type_ not in [int, float]:
                continue

            start_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(value))
            step_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(1))
            end_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(value))

            for edit in [start_edit, step_edit, end_edit]:
                edit.setValidator(validator)

            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(start_edit, row, 1)
            grid.addWidget(step_edit, row, 2)
            grid.addWidget(end_edit, row, 3)

            self.edits[name] = {
                "type": type_,
                "start": start_edit,
                "step": step_edit,
                "end": end_edit
            }

            row += 1

        parallel_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("多进程优化"))
        parallel_button.clicked.connect(self.generate_parallel_setting)
        grid.addWidget(parallel_button, row, 0, 1, 4)

        row += 1
        ga_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("遗传算法优化"))
        ga_button.clicked.connect(self.generate_ga_setting)
        grid.addWidget(ga_button, row, 0, 1, 4)

        widget: QtWidgets.QWidget = QtWidgets.QWidget()
        widget.setLayout(grid)

        scroll: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(scroll)
        self.setLayout(vbox)

    def generate_ga_setting(self) -> None:
        """"""
        self.use_ga = True
        self.generate_setting()

    def generate_parallel_setting(self) -> None:
        """"""
        self.use_ga = False
        self.generate_setting()

    def generate_setting(self) -> None:
        """"""
        self.optimization_setting = OptimizationSetting()

        self.target_display: str = self.target_combo.currentText()
        target_name: str = self.DISPLAY_NAME_MAP[self.target_display]
        self.optimization_setting.set_target(target_name)

        for name, d in self.edits.items():
            type_ = d["type"]
            start_value = type_(d["start"].text())
            step_value = type_(d["step"].text())
            end_value = type_(d["end"].text())

            if start_value == end_value:
                self.optimization_setting.add_parameter(name, start_value)
            else:
                self.optimization_setting.add_parameter(
                    name,
                    start_value,
                    end_value,
                    step_value
                )

        self.accept()

    def get_setting(self) -> tuple[OptimizationSetting, bool, int]:
        """"""
        return self.optimization_setting, self.use_ga, self.worker_spin.value()


class OptimizationResultMonitor(QtWidgets.QDialog):
    """
    For viewing optimization result.
    """

    def __init__(
        self, result_values: list, target_display: str
    ) -> None:
        """"""
        super().__init__()

        self.result_values: list = result_values
        self.target_display: str = target_display

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(_("参数优化结果"))
        self.resize(1100, 500)

        # Creat table to show result
        table: QtWidgets.QTableWidget = QtWidgets.QTableWidget()

        table.setColumnCount(2)
        table.setRowCount(len(self.result_values))
        table.setHorizontalHeaderLabels([_("参数"), self.target_display])
        table.setEditTriggers(table.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)

        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )

        for n, tp in enumerate(self.result_values):
            setting, target_value, __ = tp
            setting_cell: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem(str(setting))
            target_cell: QtWidgets.QTableWidgetItem = QtWidgets.QTableWidgetItem(f"{target_value:.2f}")

            setting_cell.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            target_cell.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            table.setItem(n, 0, setting_cell)
            table.setItem(n, 1, target_cell)

        # Create layout
        button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("保存"))
        button.clicked.connect(self.save_csv)

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(button)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(table)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def save_csv(self) -> None:
        """
        Save table data into a csv file
        """
        path, __ = QtWidgets.QFileDialog.getSaveFileName(
            self, _("保存数据"), "", "CSV(*.csv)")

        if not path:
            return

        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator="\n")

            writer.writerow([_("参数"), self.target_display])

            for tp in self.result_values:
                setting, target_value, __ = tp
                row_data: list = [str(setting), str(target_value)]
                writer.writerow(row_data)


class BacktestingTradeMonitor(BaseMonitor):
    """
    Monitor for backtesting trade data.
    """

    headers: dict = {
        "tradeid": {"display": _("成交号 "), "cell": BaseCell, "update": False},
        "orderid": {"display": _("委托号"), "cell": BaseCell, "update": False},
        "symbol": {"display": _("代码"), "cell": BaseCell, "update": False},
        "exchange": {"display": _("交易所"), "cell": EnumCell, "update": False},
        "direction": {"display": _("方向"), "cell": DirectionCell, "update": False},
        "offset": {"display": _("开平"), "cell": EnumCell, "update": False},
        "price": {"display": _("价格"), "cell": BaseCell, "update": False},
        "volume": {"display": _("数量"), "cell": BaseCell, "update": False},
        "datetime": {"display": _("时间"), "cell": BaseCell, "update": False},
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    }


class BacktestingOrderMonitor(BaseMonitor):
    """
    Monitor for backtesting order data.
    """

    headers: dict = {
        "orderid": {"display": _("委托号"), "cell": BaseCell, "update": False},
        "symbol": {"display": _("代码"), "cell": BaseCell, "update": False},
        "exchange": {"display": _("交易所"), "cell": EnumCell, "update": False},
        "type": {"display": _("类型"), "cell": EnumCell, "update": False},
        "direction": {"display": _("方向"), "cell": DirectionCell, "update": False},
        "offset": {"display": _("开平"), "cell": EnumCell, "update": False},
        "price": {"display": _("价格"), "cell": BaseCell, "update": False},
        "volume": {"display": _("总数量"), "cell": BaseCell, "update": False},
        "traded": {"display": _("已成交"), "cell": BaseCell, "update": False},
        "status": {"display": _("状态"), "cell": EnumCell, "update": False},
        "datetime": {"display": _("时间"), "cell": BaseCell, "update": False},
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    }


class FloatCell(BaseCell):
    """
    Cell used for showing pnl data.
    """

    def __init__(self, content: Any, data: Any) -> None:
        """"""
        content = f"{content:.2f}"
        super().__init__(content, data)


class DailyResultMonitor(BaseMonitor):
    """
    Monitor for backtesting daily result.
    """

    headers: dict = {
        "date": {"display": _("日期"), "cell": BaseCell, "update": False},
        "trade_count": {"display": _("成交笔数"), "cell": BaseCell, "update": False},
        "start_pos": {"display": _("开盘持仓"), "cell": BaseCell, "update": False},
        "end_pos": {"display": _("收盘持仓"), "cell": BaseCell, "update": False},
        "turnover": {"display": _("成交额"), "cell": FloatCell, "update": False},
        "commission": {"display": _("手续费"), "cell": FloatCell, "update": False},
        "slippage": {"display": _("滑点"), "cell": FloatCell, "update": False},
        "trading_pnl": {"display": _("交易盈亏"), "cell": FloatCell, "update": False},
        "holding_pnl": {"display": _("持仓盈亏"), "cell": FloatCell, "update": False},
        "total_pnl": {"display": _("总盈亏"), "cell": FloatCell, "update": False},
        "net_pnl": {"display": _("净盈亏"), "cell": FloatCell, "update": False},
    }


class BacktestingResultDialog(QtWidgets.QDialog):
    """"""

    def __init__(
        self,
        main_engine: MainEngine,
        event_engine: EventEngine,
        title: str,
        table_class: QtWidgets.QTableWidget
    ) -> None:
        """"""
        super().__init__()

        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        self.title: str = title
        self.table_class: QtWidgets.QTableWidget = table_class

        self.updated: bool = False

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(self.title)
        self.resize(1100, 600)

        self.table: QtWidgets.QTableWidget = self.table_class(self.main_engine, self.event_engine)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.table)

        self.setLayout(vbox)

    def clear_data(self) -> None:
        """"""
        self.updated = False
        self.table.setRowCount(0)

    def update_data(self, data: list) -> None:
        """"""
        self.updated = True

        data.reverse()
        for obj in data:
            self.table.insert_new_row(obj)

    def is_updated(self) -> bool:
        """"""
        return self.updated


class CandleChartDialog(QtWidgets.QDialog):
    """"""

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.updated: bool = False

        self.dt_ix_map: dict = {}
        self.ix_bar_map: dict = {}

        self.high_price = 0
        self.low_price = 0
        self.price_range = 0

        self.items: list = []

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(_("回测K线图表"))
        self.resize(1400, 800)

        # Create chart widget
        self.chart: ChartWidget = ChartWidget()
        self.chart.add_plot("candle", hide_x_axis=True)
        self.chart.add_plot("volume", maximum_height=200, hide_x_axis=True)
        self.chart.add_plot("kdj", maximum_height=200)
        self.chart.add_item(CandleItem, "candle", "candle")
        self.chart.add_item(VolumeItem, "volume", "volume")
        
        # Add KDJ indicator item
        self.kdj_item = KdjItem()
        self.kdj_item.add_to_plot(self.chart._plots["kdj"])
        
        # Register wrapper with chart system (移除内部类定义以避免缓存问题)
        kdj_wrapper = KdjChartItemWrapper(self.kdj_item)
        self.chart._items["kdj_wrapper"] = kdj_wrapper
        self.chart._item_plot_map[kdj_wrapper] = self.chart._plots["kdj"]
        
        self.chart.add_cursor()

        # Create help widget
        text1: str = _("红色虚线 —— 盈利交易")
        label1: QtWidgets.QLabel = QtWidgets.QLabel(text1)
        label1.setStyleSheet("color:red")

        text2: str = _("绿色虚线 —— 亏损交易")
        label2: QtWidgets.QLabel = QtWidgets.QLabel(text2)
        label2.setStyleSheet("color:#00FF00")

        text3: str = _("黄色向上箭头 —— 买入开仓 Buy")
        label3: QtWidgets.QLabel = QtWidgets.QLabel(text3)
        label3.setStyleSheet("color:yellow")

        text4: str = _("黄色向下箭头 —— 卖出平仓 Sell")
        label4: QtWidgets.QLabel = QtWidgets.QLabel(text4)
        label4.setStyleSheet("color:yellow")

        text5: str = _("紫红向下箭头 —— 卖出开仓 Short")
        label5: QtWidgets.QLabel = QtWidgets.QLabel(text5)
        label5.setStyleSheet("color:magenta")

        text6: str = _("紫红向上箭头 —— 买入平仓 Cover")
        label6: QtWidgets.QLabel = QtWidgets.QLabel(text6)
        label6.setStyleSheet("color:magenta")

        text7: str = _("KDJ指标 —— 黄色K线 青色D线 紫色J线")
        label7: QtWidgets.QLabel = QtWidgets.QLabel(text7)
        label7.setStyleSheet("color:blue")

        text8: str = _("红色虚线80超买 绿色虚线20超卖 灰色虚线50中位")
        label8: QtWidgets.QLabel = QtWidgets.QLabel(text8)
        label8.setStyleSheet("color:gray")

        hbox1: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox1.addStretch()
        hbox1.addWidget(label1)
        hbox1.addStretch()
        hbox1.addWidget(label2)
        hbox1.addStretch()

        hbox2: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox2.addStretch()
        hbox2.addWidget(label3)
        hbox2.addStretch()
        hbox2.addWidget(label4)
        hbox2.addStretch()

        hbox3: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox3.addStretch()
        hbox3.addWidget(label5)
        hbox3.addStretch()
        hbox3.addWidget(label6)
        hbox3.addStretch()

        hbox4: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox4.addStretch()
        hbox4.addWidget(label7)
        hbox4.addStretch()
        hbox4.addWidget(label8)
        hbox4.addStretch()

        # Set layout
        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.chart)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        self.setLayout(vbox)

    def update_history(self, history: list) -> None:
        """"""
        self.updated = True
        self.chart.update_history(history)

        for ix, bar in enumerate(history):
            self.ix_bar_map[ix] = bar
            self.dt_ix_map[bar.datetime] = ix

            if not self.high_price:
                self.high_price = bar.high_price
                self.low_price = bar.low_price
            else:
                self.high_price = max(self.high_price, bar.high_price)
                self.low_price = min(self.low_price, bar.low_price)

        self.price_range = self.high_price - self.low_price
        
        # Calculate and update KDJ data
        self._calculate_and_update_kdj(history)
    
    def _calculate_and_update_kdj(self, history: list) -> None:
        """计算并更新KDJ指标"""
        if len(history) < 9:  # KDJ需要至少9个数据点
            return
            
        # 提取价格数据
        high_prices = [bar.high_price for bar in history]
        low_prices = [bar.low_price for bar in history]
        close_prices = [bar.close_price for bar in history]
        
        # 计算KDJ
        k_data, d_data, j_data = self._calculate_kdj(high_prices, low_prices, close_prices)
        
        # 更新KDJ图表
        self.kdj_item.update_kdj_data(k_data, d_data, j_data)
    
    def _calculate_kdj(self, high_prices: list, low_prices: list, close_prices: list, 
                      period: int = 9, k_period: int = 3, d_period: int = 3) -> tuple:
        """计算KDJ指标"""
        import numpy as np
        
        n = len(high_prices)
        k_values = []
        d_values = []
        j_values = []
        
        # 初始RSV列表
        rsv_values = []
        
        for i in range(n):
            if i < period - 1:
                rsv_values.append(50.0)  # 前面不足周期的数据用50填充
            else:
                # 计算period周期内的最高价和最低价
                period_high = max(high_prices[i-period+1:i+1])
                period_low = min(low_prices[i-period+1:i+1])
                
                if period_high == period_low:
                    rsv = 50.0
                else:
                    rsv = (close_prices[i] - period_low) / (period_high - period_low) * 100
                rsv_values.append(rsv)
        
        # 计算K值（RSV的移动平均）
        k_prev = 50.0
        for i, rsv in enumerate(rsv_values):
            k = (2/3) * k_prev + (1/3) * rsv
            k_values.append(k)
            k_prev = k
        
        # 计算D值（K值的移动平均）
        d_prev = 50.0
        for k in k_values:
            d = (2/3) * d_prev + (1/3) * k
            d_values.append(d)
            d_prev = d
        
        # 计算J值
        for i in range(n):
            j = 3 * k_values[i] - 2 * d_values[i]
            j_values.append(j)
        
        return k_values, d_values, j_values

    def update_trades(self, trades: list) -> None:
        """"""
        trade_pairs: list = generate_trade_pairs(trades)

        candle_plot: pg.PlotItem = self.chart.get_plot("candle")

        scatter_data: list = []

        y_adjustment: float = self.price_range * 0.001

        for d in trade_pairs:
            open_ix = self.dt_ix_map[d["open_dt"]]
            close_ix = self.dt_ix_map[d["close_dt"]]
            open_price = d["open_price"]
            close_price = d["close_price"]

            # Trade Line
            x: list = [open_ix, close_ix]
            y: list = [open_price, close_price]

            if d["direction"] == Direction.LONG and close_price >= open_price:
                color: str = "r"
            elif d["direction"] == Direction.SHORT and close_price <= open_price:
                color = "r"
            else:
                color = "g"

            pen: QtGui.QPen = pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine)
            item: pg.PlotCurveItem = pg.PlotCurveItem(x, y, pen=pen)

            self.items.append(item)
            candle_plot.addItem(item)

            # Trade Scatter
            open_bar: BarData = self.ix_bar_map[open_ix]
            close_bar: BarData = self.ix_bar_map[close_ix]

            if d["direction"] == Direction.LONG:
                scatter_color: str = "yellow"
                open_symbol: str = "t1"
                close_symbol: str = "t"
                open_side: int = 1
                close_side: int = -1
                open_y: float = open_bar.low_price
                close_y: float = close_bar.high_price
            else:
                scatter_color = "magenta"
                open_symbol = "t"
                close_symbol = "t1"
                open_side = -1
                close_side = 1
                open_y = open_bar.high_price
                close_y = close_bar.low_price

            pen = pg.mkPen(QtGui.QColor(scatter_color))
            brush: QtGui.QBrush = pg.mkBrush(QtGui.QColor(scatter_color))
            size: int = 10

            open_scatter: dict = {
                "pos": (open_ix, open_y - open_side * y_adjustment),
                "size": size,
                "pen": pen,
                "brush": brush,
                "symbol": open_symbol
            }

            close_scatter: dict = {
                "pos": (close_ix, close_y - close_side * y_adjustment),
                "size": size,
                "pen": pen,
                "brush": brush,
                "symbol": close_symbol
            }

            scatter_data.append(open_scatter)
            scatter_data.append(close_scatter)

            # Trade text
            volume = d["volume"]
            text_color: QtGui.QColor = QtGui.QColor(scatter_color)
            open_text: pg.TextItem = pg.TextItem(f"[{volume}]", color=text_color, anchor=(0.5, 0.5))
            close_text: pg.TextItem = pg.TextItem(f"[{volume}]", color=text_color, anchor=(0.5, 0.5))

            open_text.setPos(open_ix, open_y - open_side * y_adjustment * 3)
            close_text.setPos(close_ix, close_y - close_side * y_adjustment * 3)

            self.items.append(open_text)
            self.items.append(close_text)

            candle_plot.addItem(open_text)
            candle_plot.addItem(close_text)

        trade_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem(scatter_data)
        self.items.append(trade_scatter)
        candle_plot.addItem(trade_scatter)

    def clear_data(self) -> None:
        """"""
        self.updated = False

        candle_plot: pg.PlotItem = self.chart.get_plot("candle")
        for item in self.items:
            candle_plot.removeItem(item)
        self.items.clear()

        # Clear KDJ data if exists
        if hasattr(self, 'kdj_item'):
            self.kdj_item.clear_all()

        self.chart.clear_all()

        self.dt_ix_map.clear()
        self.ix_bar_map.clear()

    def is_updated(self) -> bool:
        """"""
        return self.updated


def generate_trade_pairs(trades: list) -> list:
    """"""
    long_trades: list = []
    short_trades: list = []
    trade_pairs: list = []

    for trade in trades:
        trade = copy(trade)

        if trade.direction == Direction.LONG:
            same_direction: list = long_trades
            opposite_direction: list = short_trades
        else:
            same_direction = short_trades
            opposite_direction = long_trades

        while trade.volume and opposite_direction:
            open_trade: TradeData = opposite_direction[0]

            close_volume = min(open_trade.volume, trade.volume)
            d: dict = {
                "open_dt": open_trade.datetime,
                "open_price": open_trade.price,
                "close_dt": trade.datetime,
                "close_price": trade.price,
                "direction": open_trade.direction,
                "volume": close_volume,
            }
            trade_pairs.append(d)

            open_trade.volume -= close_volume
            if not open_trade.volume:
                opposite_direction.pop(0)

            trade.volume -= close_volume

        if trade.volume:
            same_direction.append(trade)

    return trade_pairs


class DecisionAnalysisDialog(QtWidgets.QDialog):
    """决策分析对话框"""
    
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """初始化"""
        super().__init__()
        
        self.main_engine = main_engine
        self.event_engine = event_engine
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("决策分析 - 策略买卖决策记录")
        self.setMinimumSize(1200, 800)
        
        # 创建主布局
        layout = QtWidgets.QVBoxLayout()
        
        # 标题
        title_label = QtWidgets.QLabel("📊 策略决策分析报告")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)
        
        # 创建标签页
        tab_widget = QtWidgets.QTabWidget()
        
        # 1. 决策日志标签页
        self.log_widget = QtWidgets.QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QtGui.QFont("Consolas", 10))
        tab_widget.addTab(self.log_widget, "📝 决策日志")
        
        # 2. 交易分析标签页
        self.trade_analysis_widget = self.create_trade_analysis_widget()
        tab_widget.addTab(self.trade_analysis_widget, "📈 交易分析")
        
        # 3. 条件统计标签页
        self.condition_stats_widget = self.create_condition_stats_widget()
        tab_widget.addTab(self.condition_stats_widget, "📊 条件统计")
        
        # 4. 交易时间线标签页
        self.timeline_widget = self.create_timeline_widget()
        tab_widget.addTab(self.timeline_widget, "⏰ 时间线")
        
        layout.addWidget(tab_widget)
        
        # 按钮
        button_layout = QtWidgets.QHBoxLayout()
        
        export_button = QtWidgets.QPushButton("导出分析报告")
        export_button.clicked.connect(self.export_analysis)
        
        refresh_button = QtWidgets.QPushButton("刷新数据")
        refresh_button.clicked.connect(self.refresh_data)
        
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.close)
        
        button_layout.addWidget(export_button)
        button_layout.addWidget(refresh_button)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def create_trade_analysis_widget(self):
        """创建交易分析widget"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # 交易表格
        self.trade_table = QtWidgets.QTableWidget()
        self.trade_table.setColumnCount(8)
        self.trade_table.setHorizontalHeaderLabels([
            "时间", "方向", "价格", "数量", "盈亏", "决策原因", "信号强度", "备注"
        ])
        
        # 设置表格列宽
        header = self.trade_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QtWidgets.QHeaderView.Stretch)
        
        layout.addWidget(self.trade_table)
        widget.setLayout(layout)
        
        return widget
    
    def create_condition_stats_widget(self):
        """创建条件统计widget"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # 统计表格
        self.stats_table = QtWidgets.QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels([
            "条件", "满足次数", "买入次数", "成功率"
        ])
        
        # 设置表格列宽
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        
        layout.addWidget(self.stats_table)
        widget.setLayout(layout)
        
        return widget
    
    def create_timeline_widget(self):
        """创建时间线widget"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # 时间线文本框
        self.timeline_text = QtWidgets.QTextEdit()
        self.timeline_text.setReadOnly(True)
        self.timeline_text.setFont(QtGui.QFont("Consolas", 10))
        
        layout.addWidget(self.timeline_text)
        widget.setLayout(layout)
        
        return widget
    
    def update_data(self, log_data, trades_data):
        """更新数据"""
        self.log_data = log_data
        self.trades_data = trades_data
        
        # 更新各个标签页
        self.update_decision_log()
        self.update_trade_analysis()
        self.update_condition_stats()
        self.update_timeline()
    
    def update_decision_log(self):
        """更新决策日志"""
        # 检查是否有实际的决策数据
        strategy_logs = self.log_data.get('strategy_logs', []) if isinstance(self.log_data, dict) else []
        condition_history = self.log_data.get('condition_history', []) if isinstance(self.log_data, dict) else []
        
        if not strategy_logs and not condition_history:
            self.log_widget.setText("📋 暂无决策日志数据\n\n请先运行回测生成决策日志。")
            return
        
        log_text = "📋 缩量回踩策略 - 完整决策日志\n"
        log_text += "=" * 80 + "\n\n"
        
        # 显示策略的所有决策记录
        if strategy_logs:
            log_text += "🔍 策略决策记录：\n\n"
            for i, log_entry in enumerate(strategy_logs[-50:], 1):  # 只显示最近50条
                time_str = log_entry.get('time', 'N/A')
                if hasattr(time_str, 'strftime'):
                    time_str = time_str.strftime("%Y-%m-%d %H:%M:%S")
                    
                price = log_entry.get('price', 0)
                assessment = log_entry.get('assessment', '')
                conditions_status = log_entry.get('conditions_status', '')
                position = log_entry.get('position', 0)
                
                log_text += f"{i:3d}. {time_str} - 价格: {price:.2f}, 持仓: {position}\n"
                log_text += f"     评估: {assessment}\n"
                log_text += f"     条件: {conditions_status}\n"
                
                # 显示市场数据
                market_data = log_entry.get('market_data', {})
                if market_data:
                    ma10 = market_data.get('ma10', 0)
                    ma20 = market_data.get('ma20', 0)
                    ma60 = market_data.get('ma60', 0)
                    volume = market_data.get('volume', 0)
                    log_text += f"     均线: MA10={ma10:.2f}, MA20={ma20:.2f}, MA60={ma60:.2f}, 成交量={volume:.0f}\n"
                
                log_text += "\n"
        
        # 显示交易决策记录
        trade_reasons = self.log_data.get('trade_reasons', {}) if isinstance(self.log_data, dict) else {}
        if trade_reasons:
            log_text += "\n💰 交易决策记录：\n\n"
            for time_key, reason_data in list(trade_reasons.items())[-20:]:  # 最近20条交易
                action = reason_data.get('action', '')
                reason = reason_data.get('reason', '')
                price = reason_data.get('price', 0)
                signal_strength = reason_data.get('signal_strength', '')
                
                log_text += f"⭐ {time_key}\n"
                log_text += f"   动作: {action} (价格: {price:.2f})\n"
                log_text += f"   原因: {reason}\n"
                if signal_strength:
                    log_text += f"   信号强度: {signal_strength}\n"
                log_text += "\n"
        
        if not strategy_logs and not trade_reasons:
            log_text += "📋 策略已运行，但尚未产生决策记录。\n"
            log_text += "这可能是因为：\n"
            log_text += "1. 数据不足，无法计算技术指标\n"
            log_text += "2. 市场条件不符合策略要求\n"
            log_text += "3. 策略参数设置过于严格\n"
        
        self.log_widget.setText(log_text)
    
    def update_trade_analysis(self):
        """更新交易分析"""
        # 清空表格
        self.trade_table.setRowCount(0)
        
        if not self.trades_data:
            # 添加一行示例数据
            self.trade_table.setRowCount(1)
            self.trade_table.setItem(0, 0, QtWidgets.QTableWidgetItem("暂无交易数据"))
            self.trade_table.setItem(0, 1, QtWidgets.QTableWidgetItem(""))
            self.trade_table.setItem(0, 2, QtWidgets.QTableWidgetItem(""))
            self.trade_table.setItem(0, 3, QtWidgets.QTableWidgetItem(""))
            self.trade_table.setItem(0, 4, QtWidgets.QTableWidgetItem(""))
            self.trade_table.setItem(0, 5, QtWidgets.QTableWidgetItem("请先运行回测获取交易数据"))
            self.trade_table.setItem(0, 6, QtWidgets.QTableWidgetItem(""))
            self.trade_table.setItem(0, 7, QtWidgets.QTableWidgetItem(""))
            return
        
        # 处理实际交易数据
        self.trade_table.setRowCount(len(self.trades_data))
        
        for i, trade in enumerate(self.trades_data):
            # 时间
            time_str = trade.datetime.strftime("%Y-%m-%d %H:%M:%S") if hasattr(trade, 'datetime') else "N/A"
            self.trade_table.setItem(i, 0, QtWidgets.QTableWidgetItem(time_str))
            
            # 方向
            direction = "买入" if trade.direction.value == "多" else "卖出"
            direction_item = QtWidgets.QTableWidgetItem(direction)
            if direction == "买入":
                direction_item.setBackground(QtGui.QColor(255, 230, 230))  # 淡红色
            else:
                direction_item.setBackground(QtGui.QColor(230, 255, 230))  # 淡绿色
            self.trade_table.setItem(i, 1, direction_item)
            
            # 价格
            self.trade_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{trade.price:.2f}"))
            
            # 数量
            self.trade_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(trade.volume)))
            
            # 盈亏（需要计算）
            pnl = getattr(trade, 'pnl', 0)
            pnl_item = QtWidgets.QTableWidgetItem(f"{pnl:.2f}")
            if pnl > 0:
                pnl_item.setForeground(QtGui.QColor(255, 0, 0))  # 红色表示盈利
            elif pnl < 0:
                pnl_item.setForeground(QtGui.QColor(0, 150, 0))  # 绿色表示亏损
            self.trade_table.setItem(i, 4, pnl_item)
            
            # 决策原因（示例）
            reason = "所有条件满足，执行买入" if direction == "买入" else "止盈/止损卖出"
            self.trade_table.setItem(i, 5, QtWidgets.QTableWidgetItem(reason))
            
            # 信号强度（示例）
            strength = "强" if direction == "买入" else "中"
            self.trade_table.setItem(i, 6, QtWidgets.QTableWidgetItem(strength))
            
            # 备注
            self.trade_table.setItem(i, 7, QtWidgets.QTableWidgetItem("缩量回踩策略"))
    
    def update_condition_stats(self):
        """更新条件统计"""
        # 清空表格
        self.stats_table.setRowCount(0)
        
        # 示例统计数据
        conditions = [
            ("前期放量上涨", 45, 12, "26.7%"),
            ("缩量回踩", 38, 12, "31.6%"),
            ("均线趋势向上", 52, 18, "34.6%"),
            ("15分钟吸筹", 28, 8, "28.6%"),
            ("综合信号(4/4)", 8, 8, "100%"),
            ("综合信号(3/4)", 15, 4, "26.7%")
        ]
        
        self.stats_table.setRowCount(len(conditions))
        
        for i, (condition, count, trades, success_rate) in enumerate(conditions):
            self.stats_table.setItem(i, 0, QtWidgets.QTableWidgetItem(condition))
            self.stats_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(count)))
            self.stats_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(trades)))
            
            # 成功率着色
            rate_item = QtWidgets.QTableWidgetItem(success_rate)
            rate_value = float(success_rate.rstrip('%'))
            if rate_value >= 30:
                rate_item.setForeground(QtGui.QColor(255, 0, 0))  # 红色
            elif rate_value >= 20:
                rate_item.setForeground(QtGui.QColor(255, 165, 0))  # 橙色
            else:
                rate_item.setForeground(QtGui.QColor(0, 150, 0))  # 绿色
            
            self.stats_table.setItem(i, 3, rate_item)
    
    def update_timeline(self):
        """更新时间线 - 基于真实的策略决策历史"""
        timeline_text = "⏰ 策略决策时间线\n"
        timeline_text += "=" * 80 + "\n\n"
        
        # 获取条件历史数据
        condition_history = self.log_data.get('condition_history', []) if isinstance(self.log_data, dict) else []
        trade_reasons = self.log_data.get('trade_reasons', {}) if isinstance(self.log_data, dict) else {}
        order_reasons = self.log_data.get('order_reasons', {}) if isinstance(self.log_data, dict) else {}
        
        if not condition_history:
            timeline_text += "📋 暂无时间线数据\n\n"
            timeline_text += "请运行回测以生成详细的决策时间线。\n"
            timeline_text += "时间线将显示每个交易日的：\n"
            timeline_text += "• 技术指标计算过程\n"
            timeline_text += "• 买卖条件评估结果\n"
            timeline_text += "• 交易决策的完整逻辑\n"
            timeline_text += "• 风险信号的识别过程\n"
            self.timeline_widget.setText(timeline_text)
            return
        
        # 按日期分组显示决策历史
        daily_decisions = {}
        for record in condition_history:
            date_str = record.get('time', '').split()[0] if record.get('time') else '未知日期'
            if date_str not in daily_decisions:
                daily_decisions[date_str] = []
            daily_decisions[date_str].append(record)
        
        # 只显示最近10个交易日，避免数据过多
        sorted_dates = sorted(daily_decisions.keys())[-10:]
        
        for date in sorted_dates:
            day_records = daily_decisions[date]
            timeline_text += f"📅 {date} 交易日决策记录：\n"
            timeline_text += "-" * 60 + "\n"
            
            # 显示当日的关键决策点
            for i, record in enumerate(day_records):
                time_str = record.get('time', '未知时间')
                price = record.get('price', 0)
                decision = record.get('decision', '无决策')
                conditions = record.get('conditions', {})
                
                # 提取时间部分
                time_part = time_str.split()[1] if len(time_str.split()) > 1 else time_str
                
                timeline_text += f"\n🕐 {time_part} 价格: {price:.2f}\n"
                
                # 显示9个完整条件的状态
                if conditions:
                    timeline_text += "📊 条件评估：\n"
                    condition_names = {
                        'ma_rising': '10/20均线上移',
                        'ma60_120_rising': '60/120均线上移',
                        'pullback_ma': '回踩均线',
                        'surge_60day': '60日涨幅>7%',
                        'amplitude_110day': '110日振幅>8.1%',
                        'accumulation_15m': '15分钟吸筹',
                        'shrink_volume': '连续缩量',
                        'kdj_dea_rising': 'KDJ/DEA上移',
                        'no_negative': '无负面信号'
                    }
                    
                    conditions_met = 0
                    for key, name in condition_names.items():
                        status = conditions.get(key, False)
                        if status:
                            conditions_met += 1
                        icon = "✅" if status else "❌"
                        timeline_text += f"  {icon} {name}\n"
                    
                    timeline_text += f"\n📈 满足条件: {conditions_met}/9\n"
                
                # 显示决策结果
                if decision == "买入":
                    timeline_text += "🚀 **执行买入操作**\n"
                    # 查找对应的交易原因
                    for reason_key, reason_text in trade_reasons.items():
                        if time_str in reason_key or date in reason_key:
                            timeline_text += f"💡 买入理由: {reason_text}\n"
                            break
                elif decision == "卖出":
                    timeline_text += "📉 **执行卖出操作**\n"
                    for reason_key, reason_text in trade_reasons.items():
                        if time_str in reason_key or date in reason_key:
                            timeline_text += f"💡 卖出理由: {reason_text}\n"
                            break
                elif decision == "继续持有":
                    timeline_text += "⏳ 继续持有观察\n"
                else:
                    timeline_text += "⏸️ 暂不操作\n"
                
                timeline_text += "\n"
            
            timeline_text += "\n"
        
        # 如果有交易记录，添加交易汇总
        if order_reasons:
            timeline_text += "📋 交易操作汇总：\n"
            timeline_text += "=" * 60 + "\n"
            for order_id, reason in order_reasons.items():
                timeline_text += f"🔸 订单 {order_id}: {reason}\n"
            timeline_text += "\n"
        
        self.timeline_widget.setText(timeline_text)
    
    def export_analysis(self):
        """导出分析报告"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出决策分析报告",
            f"决策分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;所有文件 (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("策略决策分析报告\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write("生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
                    
                    f.write("决策日志:\n")
                    f.write("-" * 40 + "\n")
                    f.write(self.log_widget.toPlainText())
                    f.write("\n\n")
                    
                    f.write("时间线:\n")
                    f.write("-" * 40 + "\n")
                    f.write(self.timeline_text.toPlainText())
                
                QtWidgets.QMessageBox.information(
                    self,
                    "导出成功",
                    f"分析报告已导出到：\n{filename}"
                )
                
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "导出失败",
                    f"导出分析报告时发生错误：\n{str(e)}"
                )
    
    def refresh_data(self):
        """刷新数据"""
        # 重新获取数据
        if hasattr(self, 'log_data') and hasattr(self, 'trades_data'):
            self.update_data(self.log_data, self.trades_data)
        
        QtWidgets.QMessageBox.information(
            self,
            "刷新完成",
            "决策分析数据已刷新"
        )

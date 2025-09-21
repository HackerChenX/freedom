"""
数据查看器UI界面
遵循VnPy UI架构设计
"""

from typing import List
from datetime import datetime, timedelta

from vnpy.trader.ui import QtWidgets, QtCore, QtGui
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData

from .engine import DataViewerEngine


class DataViewerWidget(QtWidgets.QWidget):
    """数据查看器主界面"""
    
    def __init__(self, engine: DataViewerEngine):
        """初始化界面"""
        super().__init__()
        
        self.engine = engine
        
        self.init_ui()
        self.load_data_overview()
    
    def init_ui(self) -> None:
        """初始化界面"""
        self.setWindowTitle("数据查看器")
        self.resize(1200, 800)
        
        # 创建主布局
        layout = QtWidgets.QVBoxLayout()
        
        # 创建标签页
        self.tab_widget = QtWidgets.QTabWidget()
        
        # K线数据标签页
        self.bar_widget = BarDataWidget(self.engine)
        self.tab_widget.addTab(self.bar_widget, "K线数据")
        
        # Tick数据标签页
        self.tick_widget = TickDataWidget(self.engine)
        self.tab_widget.addTab(self.tick_widget, "Tick数据")
        
        # 数据概览标签页
        self.overview_widget = OverviewWidget(self.engine)
        self.tab_widget.addTab(self.overview_widget, "数据概览")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def load_data_overview(self) -> None:
        """加载数据概览"""
        self.engine.get_bar_overview()
        self.engine.get_tick_overview()
        
        # 刷新各个标签页
        self.bar_widget.refresh_symbols()
        self.tick_widget.refresh_symbols()
        self.overview_widget.refresh_overview()


class BarDataWidget(QtWidgets.QWidget):
    """K线数据查看界面"""
    
    def __init__(self, engine: DataViewerEngine):
        """初始化"""
        super().__init__()
        
        self.engine = engine
        self.bars: List[BarData] = []
        
        self.init_ui()
    
    def init_ui(self) -> None:
        """初始化界面"""
        layout = QtWidgets.QVBoxLayout()
        
        # 查询控制面板
        control_layout = QtWidgets.QHBoxLayout()
        
        # 合约选择
        control_layout.addWidget(QtWidgets.QLabel("合约:"))
        self.symbol_combo = QtWidgets.QComboBox()
        self.symbol_combo.setMinimumWidth(150)
        self.symbol_combo.currentTextChanged.connect(self.on_symbol_changed)
        control_layout.addWidget(self.symbol_combo)
        
        # 时间周期选择
        control_layout.addWidget(QtWidgets.QLabel("周期:"))
        self.interval_combo = QtWidgets.QComboBox()
        self.interval_combo.setMinimumWidth(100)
        control_layout.addWidget(self.interval_combo)
        
        # 开始时间
        control_layout.addWidget(QtWidgets.QLabel("开始:"))
        self.start_edit = QtWidgets.QDateTimeEdit()
        self.start_edit.setDateTime(QtCore.QDateTime.currentDateTime().addDays(-30))
        self.start_edit.setCalendarPopup(True)
        control_layout.addWidget(self.start_edit)
        
        # 结束时间
        control_layout.addWidget(QtWidgets.QLabel("结束:"))
        self.end_edit = QtWidgets.QDateTimeEdit()
        self.end_edit.setDateTime(QtCore.QDateTime.currentDateTime())
        self.end_edit.setCalendarPopup(True)
        control_layout.addWidget(self.end_edit)
        
        # 查询按钮
        query_button = QtWidgets.QPushButton("查询")
        query_button.clicked.connect(self.query_data)
        control_layout.addWidget(query_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 数据表格
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "时间", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "持仓量"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        
        # 状态栏
        self.status_label = QtWidgets.QLabel("就绪")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def refresh_symbols(self) -> None:
        """刷新合约列表"""
        symbols = self.engine.get_available_symbols()
        self.symbol_combo.clear()
        self.symbol_combo.addItems(symbols)
    
    def on_symbol_changed(self) -> None:
        """合约变化时更新时间周期"""
        symbol_text = self.symbol_combo.currentText()
        if not symbol_text:
            return
        
        try:
            symbol, exchange_str = symbol_text.split(".")
            exchange = Exchange(exchange_str)
            
            intervals = self.engine.get_available_intervals(symbol, exchange)
            self.interval_combo.clear()
            
            for interval in intervals:
                self.interval_combo.addItem(interval.value, interval)
            
            # 更新时间范围
            if intervals:
                start_time, end_time = self.engine.get_data_range(symbol, exchange, intervals[0])
                if start_time and end_time:
                    self.start_edit.setDateTime(QtCore.QDateTime(start_time))
                    self.end_edit.setDateTime(QtCore.QDateTime(end_time))
        
        except Exception as e:
            self.status_label.setText(f"更新时间周期失败: {e}")
    
    def query_data(self) -> None:
        """查询数据"""
        symbol_text = self.symbol_combo.currentText()
        if not symbol_text:
            self.status_label.setText("请选择合约")
            return
        
        try:
            symbol, exchange_str = symbol_text.split(".")
            exchange = Exchange(exchange_str)
            interval = self.interval_combo.currentData()
            
            start = self.start_edit.dateTime().toPython()
            end = self.end_edit.dateTime().toPython()
            
            self.status_label.setText("正在查询数据...")
            QtWidgets.QApplication.processEvents()
            
            # 查询数据
            self.bars = self.engine.load_bar_data(symbol, exchange, interval, start, end)
            
            # 显示数据
            self.show_data()
            
            self.status_label.setText(f"查询完成，共 {len(self.bars)} 条数据")
        
        except Exception as e:
            self.status_label.setText(f"查询失败: {e}")
    
    def show_data(self) -> None:
        """显示数据"""
        self.table.setRowCount(len(self.bars))
        
        for row, bar in enumerate(self.bars):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(bar.datetime.strftime("%Y-%m-%d %H:%M:%S")))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{bar.open_price:.2f}"))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{bar.high_price:.2f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{bar.low_price:.2f}"))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{bar.close_price:.2f}"))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{bar.volume:,.0f}"))
            self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(f"{bar.turnover:,.0f}"))
            self.table.setItem(row, 7, QtWidgets.QTableWidgetItem(f"{bar.open_interest:,.0f}"))


class TickDataWidget(QtWidgets.QWidget):
    """Tick数据查看界面"""
    
    def __init__(self, engine: DataViewerEngine):
        """初始化"""
        super().__init__()
        
        self.engine = engine
        self.ticks: List[TickData] = []
        
        self.init_ui()
    
    def init_ui(self) -> None:
        """初始化界面"""
        layout = QtWidgets.QVBoxLayout()
        
        # 查询控制面板
        control_layout = QtWidgets.QHBoxLayout()
        
        # 合约选择
        control_layout.addWidget(QtWidgets.QLabel("合约:"))
        self.symbol_combo = QtWidgets.QComboBox()
        self.symbol_combo.setMinimumWidth(150)
        control_layout.addWidget(self.symbol_combo)
        
        # 开始时间
        control_layout.addWidget(QtWidgets.QLabel("开始:"))
        self.start_edit = QtWidgets.QDateTimeEdit()
        self.start_edit.setDateTime(QtCore.QDateTime.currentDateTime().addDays(-1))
        self.start_edit.setCalendarPopup(True)
        control_layout.addWidget(self.start_edit)
        
        # 结束时间
        control_layout.addWidget(QtWidgets.QLabel("结束:"))
        self.end_edit = QtWidgets.QDateTimeEdit()
        self.end_edit.setDateTime(QtCore.QDateTime.currentDateTime())
        self.end_edit.setCalendarPopup(True)
        control_layout.addWidget(self.end_edit)
        
        # 查询按钮
        query_button = QtWidgets.QPushButton("查询")
        query_button.clicked.connect(self.query_data)
        control_layout.addWidget(query_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 数据表格
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "时间", "最新价", "成交量", "买一价", "卖一价", "持仓量"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        
        # 状态栏
        self.status_label = QtWidgets.QLabel("就绪")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def refresh_symbols(self) -> None:
        """刷新合约列表"""
        symbols = self.engine.get_available_symbols()
        self.symbol_combo.clear()
        self.symbol_combo.addItems(symbols)
    
    def query_data(self) -> None:
        """查询数据"""
        symbol_text = self.symbol_combo.currentText()
        if not symbol_text:
            self.status_label.setText("请选择合约")
            return
        
        try:
            symbol, exchange_str = symbol_text.split(".")
            exchange = Exchange(exchange_str)
            
            start = self.start_edit.dateTime().toPython()
            end = self.end_edit.dateTime().toPython()
            
            self.status_label.setText("正在查询数据...")
            QtWidgets.QApplication.processEvents()
            
            # 查询数据
            self.ticks = self.engine.load_tick_data(symbol, exchange, start, end)
            
            # 显示数据
            self.show_data()
            
            self.status_label.setText(f"查询完成，共 {len(self.ticks)} 条数据")
        
        except Exception as e:
            self.status_label.setText(f"查询失败: {e}")
    
    def show_data(self) -> None:
        """显示数据"""
        self.table.setRowCount(len(self.ticks))
        
        for row, tick in enumerate(self.ticks):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(tick.datetime.strftime("%Y-%m-%d %H:%M:%S")))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{tick.last_price:.2f}"))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{tick.volume:,.0f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{tick.bid_price_1:.2f}"))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{tick.ask_price_1:.2f}"))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{tick.open_interest:,.0f}"))


class OverviewWidget(QtWidgets.QWidget):
    """数据概览界面"""
    
    def __init__(self, engine: DataViewerEngine):
        """初始化"""
        super().__init__()
        
        self.engine = engine
        
        self.init_ui()
    
    def init_ui(self) -> None:
        """初始化界面"""
        layout = QtWidgets.QVBoxLayout()
        
        # 刷新按钮
        refresh_button = QtWidgets.QPushButton("刷新概览")
        refresh_button.clicked.connect(self.refresh_overview)
        layout.addWidget(refresh_button)
        
        # K线概览表格
        layout.addWidget(QtWidgets.QLabel("K线数据概览:"))
        self.bar_table = QtWidgets.QTableWidget()
        self.bar_table.setColumnCount(6)
        self.bar_table.setHorizontalHeaderLabels([
            "合约", "交易所", "周期", "数据量", "开始时间", "结束时间"
        ])
        self.bar_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.bar_table)
        
        # Tick概览表格
        layout.addWidget(QtWidgets.QLabel("Tick数据概览:"))
        self.tick_table = QtWidgets.QTableWidget()
        self.tick_table.setColumnCount(5)
        self.tick_table.setHorizontalHeaderLabels([
            "合约", "交易所", "数据量", "开始时间", "结束时间"
        ])
        self.tick_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.tick_table)
        
        self.setLayout(layout)
    
    def refresh_overview(self) -> None:
        """刷新概览"""
        # 重新加载数据
        self.engine.get_bar_overview()
        self.engine.get_tick_overview()
        
        # 显示K线概览
        bar_overviews = self.engine.bar_overviews
        self.bar_table.setRowCount(len(bar_overviews))
        
        for row, overview in enumerate(bar_overviews):
            self.bar_table.setItem(row, 0, QtWidgets.QTableWidgetItem(overview.symbol))
            self.bar_table.setItem(row, 1, QtWidgets.QTableWidgetItem(overview.exchange.value))
            self.bar_table.setItem(row, 2, QtWidgets.QTableWidgetItem(overview.interval.value))
            self.bar_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{overview.count:,}"))
            self.bar_table.setItem(row, 4, QtWidgets.QTableWidgetItem(overview.start.strftime("%Y-%m-%d")))
            self.bar_table.setItem(row, 5, QtWidgets.QTableWidgetItem(overview.end.strftime("%Y-%m-%d")))
        
        # 显示Tick概览
        tick_overviews = self.engine.tick_overviews
        self.tick_table.setRowCount(len(tick_overviews))
        
        for row, overview in enumerate(tick_overviews):
            self.tick_table.setItem(row, 0, QtWidgets.QTableWidgetItem(overview.symbol))
            self.tick_table.setItem(row, 1, QtWidgets.QTableWidgetItem(overview.exchange.value))
            self.tick_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{overview.count:,}"))
            self.tick_table.setItem(row, 3, QtWidgets.QTableWidgetItem(overview.start.strftime("%Y-%m-%d")))
            self.tick_table.setItem(row, 4, QtWidgets.QTableWidgetItem(overview.end.strftime("%Y-%m-%d")))

"""
数据查看器引擎
遵循VnPy引擎架构设计，通过数据库接口访问数据
"""

from typing import List, Optional
from datetime import datetime, timedelta

from vnpy.event import Event, EventEngine
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import get_database
from vnpy.trader.utility import load_json, save_json


class DataViewerEngine(BaseEngine):
    """数据查看器引擎"""
    
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """初始化引擎"""
        super().__init__(main_engine, event_engine, "DataViewer")
        
        # 获取数据库接口（遵循VnPy架构）
        self.database = get_database()
        
        # 缓存数据
        self.bar_data_cache: List[BarData] = []
        self.tick_data_cache: List[TickData] = []
        
        # 加载配置
        self.load_setting()
    
    def load_setting(self) -> None:
        """加载配置"""
        try:
            setting = load_json("data_viewer_setting.json")
            self.setting = setting
        except FileNotFoundError:
            self.setting = {}
    
    def save_setting(self) -> None:
        """保存配置"""
        save_json("data_viewer_setting.json", self.setting)
    
    def get_bar_overview(self) -> List[dict]:
        """获取K线数据概览"""
        try:
            self.bar_overviews = []  # 暂时禁用概览功能
            self.write_log(f"获取到 {len(self.bar_overviews)} 个K线数据概览")
            return self.bar_overviews
        except Exception as e:
            self.write_log(f"获取K线概览失败: {e}")
            return []
    
    def get_tick_overview(self) -> List[dict]:
        """获取Tick数据概览"""
        try:
            self.tick_overviews = []  # 暂时禁用概览功能
            self.write_log(f"获取到 {len(self.tick_overviews)} 个Tick数据概览")
            return self.tick_overviews
        except Exception as e:
            self.write_log(f"获取Tick概览失败: {e}")
            return []
    
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """加载K线数据"""
        try:
            bars = self.database.load_bar_data(symbol, exchange, interval, start, end)
            self.write_log(f"加载 {symbol}.{exchange.value} {interval.value} 数据: {len(bars)} 条")
            return bars
        except Exception as e:
            self.write_log(f"加载K线数据失败: {e}")
            return []
    
    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """加载Tick数据"""
        try:
            ticks = self.database.load_tick_data(symbol, exchange, start, end)
            self.write_log(f"加载 {symbol}.{exchange.value} Tick数据: {len(ticks)} 条")
            return ticks
        except Exception as e:
            self.write_log(f"加载Tick数据失败: {e}")
            return []
    
    def get_available_symbols(self) -> List[str]:
        """获取可用的合约代码"""
        # 直接从数据库查询可用合约
        try:
            if hasattr(self.database, 'client'):
                # ClickHouse数据库
                result = self.database.client.query("SELECT DISTINCT code FROM stock.stock_info LIMIT 100")
                symbols = [f"{row[0]}.SSE" if row[0].startswith('6') else f"{row[0]}.SZSE" 
                          for row in result.result_rows]
                return sorted(symbols)
        except Exception:
            pass
        return []
    
    def get_available_intervals(self, symbol: str, exchange: Exchange) -> List[Interval]:
        """获取指定合约的可用时间周期"""
        intervals = []
        
        for overview in self.bar_overviews:
            if overview.symbol == symbol and overview.exchange == exchange:
                intervals.append(overview.interval)
        
        return intervals
    
    def get_data_range(self, symbol: str, exchange: Exchange, interval: Interval) -> tuple:
        """获取数据时间范围"""
        for overview in self.bar_overviews:
            if (overview.symbol == symbol and 
                overview.exchange == exchange and 
                overview.interval == interval):
                return overview.start, overview.end
        
        return None, None
    
    def close(self) -> None:
        """关闭引擎"""
        self.save_setting()
        self.write_log("数据查看器引擎已关闭")

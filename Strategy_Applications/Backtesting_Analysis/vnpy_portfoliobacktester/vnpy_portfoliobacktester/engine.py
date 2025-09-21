"""
组合策略回测引擎
"""

from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.event import EventEngine
from .backtesting import BacktestingEngine

APP_NAME = "PortfolioBacktester"


class PortfolioBacktesterEngine(BaseEngine):
    """组合策略回测引擎"""
    
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """构造函数"""
        super().__init__(main_engine, event_engine, APP_NAME)
        
        # 初始化回测引擎
        self.backtesting_engine = BacktestingEngine()
        
    def start_backtesting(self):
        """开始回测"""
        self.write_log("开始组合策略回测...")
        
    def stop_backtesting(self):
        """停止回测"""
        self.write_log("停止组合策略回测...")
        
    def get_backtesting_engine(self):
        """获取回测引擎"""
        return self.backtesting_engine

"""
双均线突破策略模块

实现基于短期均线突破长期均线的选股策略
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

from strategy.base_strategy import BaseStrategy
from formula.stock_formula import StockFormula
from utils.logger import get_logger
from indicators.ma import MA
from db.data_manager import DataManager
from models.stock_info import StockInfo  # 导入StockInfo类

logger = get_logger(__name__)


class DualMAStrategy(BaseStrategy):
    """
    双均线突破策略
    
    基于短期均线突破长期均线的选股策略，结合成交量放大等条件
    """
    
    def __init__(self):
        """初始化双均线突破策略"""
        super().__init__(name="双均线突破策略", description="基于短期均线突破长期均线的选股策略，结合成交量放大等条件")
        
        # 设置默认参数
        self._parameters = {
            'short_period': 5,       # 短期均线周期
            'long_period': 10,       # 长期均线周期
            'volume_ratio': 1.5,     # 成交量放大倍数
            'lookback_days': 3,      # 向前查找突破的天数
            'require_volume': True,  # 是否要求成交量放大
            'start_date': None,      # 起始日期，默认为None表示使用最近交易日期
            'end_date': None         # 结束日期，默认为None表示使用最近交易日期
        }
        
        # 初始化数据管理器
        self.data_manager = DataManager()
    
    def select(self, universe: List[str], *args, **kwargs) -> pd.DataFrame:
        """
        执行双均线突破选股策略
        
        Args:
            universe: 股票代码列表，表示选股范围
            args: 位置参数
            kwargs: 关键字参数，可以覆盖默认参数
            
        Returns:
            pd.DataFrame: 选股结果，包含股票代码、名称、行业等信息
        """
        # 更新参数
        if kwargs:
            self._parameters.update(kwargs)
        
        # 提取参数
        short_period = self._parameters['short_period']
        long_period = self._parameters['long_period']
        volume_ratio = self._parameters['volume_ratio']
        lookback_days = self._parameters['lookback_days']
        require_volume = self._parameters['require_volume']
        start_date = self._parameters['start_date']
        end_date = self._parameters['end_date']
        
        # 自动获取最近日期
        latest_date = self.data_manager.get_latest_trading_date()
        if end_date is None:
            end_date = latest_date
        
        if start_date is None:
            # 计算起始日期，至少要包含足够的数据计算均线
            # 默认往前推 2倍长周期 + 10 个交易日
            start_date = self.data_manager.get_previous_trading_date(end_date, long_period * 2 + 10)
        
        # 存储选股结果
        selected_stocks = []
        
        # 遍历股票池，应用选股条件
        total_stocks = len(universe)
        logger.info(f"开始对 {total_stocks} 只股票进行双均线突破策略筛选")
        
        for i, code in enumerate(universe):
            try:
                # 打印进度
                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i + 1}/{total_stocks} 只股票")
                
                # 获取股票日线数据
                df = self.data_manager.get_daily_data(code, start_date, end_date)
                
                # 跳过没有数据的股票
                if df is None or len(df) < long_period * 2:
                    continue
                
                # 计算均线
                ma_short = MA(df, short_period)
                ma_long = MA(df, long_period)
                
                # 检查是否在lookback_days内有均线突破
                has_breakout = False
                
                # 获取最近几天的数据
                recent_data = df.iloc[-lookback_days:].copy()
                recent_data['ma_short'] = ma_short[-lookback_days:]
                recent_data['ma_long'] = ma_long[-lookback_days:]
                
                # 检查每一天是否有突破
                for j in range(1, len(recent_data)):
                    # 检查前一天短期均线低于长期均线，当天短期均线高于长期均线
                    if (recent_data['ma_short'].iloc[j-1] < recent_data['ma_long'].iloc[j-1] and
                        recent_data['ma_short'].iloc[j] >= recent_data['ma_long'].iloc[j]):
                        has_breakout = True
                        
                        # 如果要求成交量放大
                        if require_volume:
                            # 计算突破当天与前N天的平均成交量比值
                            current_volume = recent_data['volume'].iloc[j]
                            avg_volume = df['volume'].iloc[-(lookback_days+5):-lookback_days].mean()
                            
                            if current_volume < avg_volume * volume_ratio:
                                has_breakout = False
                        
                        break
                
                # 如果有突破且满足条件，添加到选股结果
                if has_breakout:
                    # 获取股票基本信息
                    stock_info = self.data_manager.get_stock_info(code)
                    
                    selected_stocks.append({
                        'code': code,
                        'name': stock_info.name if stock_info else code,
                        'industry': stock_info.industry if stock_info else '',
                        'breakout_date': recent_data.index[j].strftime('%Y%m%d')
                    })
                    
                    logger.info(f"选出股票: {code} {stock_info.name if stock_info else ''}")
            
            except Exception as e:
                logger.error(f"处理股票 {code} 时出错: {e}")
        
        # 转换为DataFrame
        result_df = pd.DataFrame(selected_stocks)
        
        if len(result_df) > 0:
            # 添加排序
            result_df = result_df.sort_values(by='breakout_date', ascending=False)
        
        logger.info(f"双均线突破策略选股完成，共选出 {len(result_df)} 只股票")
        
        return result_df 
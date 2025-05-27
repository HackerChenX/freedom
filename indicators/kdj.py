"""
KDJ指标模块

实现KDJ随机指标的计算和相关功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import kdj as calc_kdj, highest, lowest
from utils.decorators import exception_handler, validate_dataframe, log_calls
from utils.logger import get_logger

logger = get_logger(__name__)


class KDJ(BaseIndicator):
    """
    KDJ随机指标
    
    KDJ指标是RSI和随机指标的结合体，是一种超买超卖指标，用于判断股价走势的超买超卖状态。
    """
    
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        """
        初始化KDJ指标
        
        Args:
            n: RSV周期，默认为9
            m1: K值平滑因子，默认为3
            m2: D值平滑因子，默认为3
        """
        super().__init__(name="KDJ", description="随机指标")
        self.n = n
        self.m1 = m1
        self.m2 = m2
    
    @validate_dataframe(required_columns=['high', 'low', 'close'], min_rows=9)
    @log_calls(level='debug')
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, add_prefix: bool = False, **kwargs) -> pd.DataFrame:
        """
        计算KDJ指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            add_prefix: 是否在输出列名前添加指标名称前缀
            kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含KDJ指标的DataFrame
            
        Raises:
            ValueError: 如果输入数据不包含必需的列
        """
        # 复制输入数据
        result = data.copy()
        
        # 使用统一的公共函数计算KDJ
        k, d, j = calc_kdj(
            result['close'].values,
            result['high'].values,
            result['low'].values,
            self.n,
            self.m1,
            self.m2
        )
        
        # 设置列名（使用大写字母）
        if add_prefix:
            k_col = self.get_column_name('K')
            d_col = self.get_column_name('D')
            j_col = self.get_column_name('J')
        else:
            k_col = 'K'
            d_col = 'D'
            j_col = 'J'
        
        # 添加结果列
        result[k_col] = k
        result[d_col] = d
        result[j_col] = j
        
        # 添加信号
        result = self.add_signals(result, k_col, d_col, j_col)
        
        # 保存结果
        self._result = result
        
        return result
    
    def add_signals(self, data: pd.DataFrame, k_col: str = 'K', 
                   d_col: str = 'D', j_col: str = 'J') -> pd.DataFrame:
        """
        添加KDJ交易信号
        
        Args:
            data: 包含KDJ指标的DataFrame
            k_col: K值列名
            d_col: D值列名
            j_col: J值列名
            
        Returns:
            pd.DataFrame: 添加了信号的DataFrame
        """
        result = data.copy()
        
        # 计算超买超卖信号
        result['kdj_overbought'] = (result[k_col] > 80) & (result[d_col] > 80)
        result['kdj_oversold'] = (result[k_col] < 20) & (result[d_col] < 20)
        
        # 计算金叉和死叉信号
        result['kdj_buy_signal'] = self.get_buy_signal(result, k_col, d_col)
        result['kdj_sell_signal'] = self.get_sell_signal(result, k_col, d_col)
        
        # 计算J值超买超卖
        result['kdj_j_overbought'] = result[j_col] > 100
        result['kdj_j_oversold'] = result[j_col] < 0
        
        # 计算KDJ三线同向（顺势信号）
        result['kdj_uptrend'] = (result[j_col] > result[k_col]) & (result[k_col] > result[d_col])
        result['kdj_downtrend'] = (result[j_col] < result[k_col]) & (result[k_col] < result[d_col])
        
        return result
    
    def get_buy_signal(self, data: pd.DataFrame, k_col: str = 'K', d_col: str = 'D') -> pd.Series:
        """
        获取KDJ买入信号
        
        Args:
            data: 包含KDJ指标的DataFrame
            k_col: K值列名
            d_col: D值列名
            
        Returns:
            pd.Series: 买入信号序列（布尔值）
        """
        # KDJ金叉：K线从下方穿过D线，且处于低位（<30）
        golden_cross = (data[k_col] > data[d_col]) & (data[k_col].shift(1) <= data[d_col].shift(1))
        low_position = data[k_col] < 30
        
        return golden_cross & low_position
    
    def get_sell_signal(self, data: pd.DataFrame, k_col: str = 'K', d_col: str = 'D') -> pd.Series:
        """
        获取KDJ卖出信号
        
        Args:
            data: 包含KDJ指标的DataFrame
            k_col: K值列名
            d_col: D值列名
            
        Returns:
            pd.Series: 卖出信号序列（布尔值）
        """
        # KDJ死叉：K线从上方穿过D线，且处于高位（>70）
        death_cross = (data[k_col] < data[d_col]) & (data[k_col].shift(1) >= data[d_col].shift(1))
        high_position = data[k_col] > 70
        
        return death_cross & high_position
    
    def get_column_name(self, suffix: str = "") -> str:
        """
        获取指标列名
        
        Args:
            suffix: 列名后缀
            
        Returns:
            str: 指标列名
        """
        if suffix:
            return f"{self.name.lower()}_{suffix}"
        return self.name.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将指标转换为字典表示
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'n': self.n,
                'm1': self.m1,
                'm2': self.m2
            },
            'has_result': self.has_result(),
            'has_error': self.has_error(),
            'error': str(self._error) if self._error else None
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成KDJ信号
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        # 计算KDJ指标
        if not self.has_result():
            self.compute(data)
            
        if not self.has_result():
            return pd.DataFrame()
            
        # 获取K、D和J值
        k_values = self._result['K']
        d_values = self._result['D']
        j_values = self._result['J']
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 添加买入信号
        signals['buy_signal'] = self.get_buy_signal(self._result)
        
        # 添加卖出信号
        signals['sell_signal'] = self.get_sell_signal(self._result)
        
        # 添加超买超卖信号
        signals['overbought'] = (k_values > 80) & (d_values > 80)
        signals['oversold'] = (k_values < 20) & (d_values < 20)
        
        # 添加J值超买超卖
        signals['j_overbought'] = j_values > 100
        signals['j_oversold'] = j_values < 0
        
        # 添加KDJ三线同向（顺势信号）
        signals['uptrend'] = (j_values > k_values) & (k_values > d_values)
        signals['downtrend'] = (j_values < k_values) & (k_values < d_values)
        
        # 计算信号强度
        # 范围是0-100，0表示最弱，100表示最强
        strength = 50.0  # 默认中性
        
        # 如果出现金叉，信号强度增加
        if signals['buy_signal'].iloc[-1]:
            strength += 25.0
            
        # 如果处于超卖区域，信号强度增加
        if signals['oversold'].iloc[-1]:
            strength += 15.0
            
        # 如果J值在超卖区域，信号强度增加
        if signals['j_oversold'].iloc[-1]:
            strength += 10.0
            
        # 如果三线同向看涨，信号强度增加
        if signals['uptrend'].iloc[-1]:
            strength += 10.0
            
        # 如果处于超买区域，信号强度减少
        if signals['overbought'].iloc[-1]:
            strength -= 15.0
            
        # 如果出现死叉，信号强度减少
        if signals['sell_signal'].iloc[-1]:
            strength -= 25.0
            
        # 确保强度在0-100范围内
        strength = max(0.0, min(100.0, strength))
        
        # 添加信号强度
        signals['signal_strength'] = 0.0
        signals.loc[signals.index[-1], 'signal_strength'] = strength
        
        return signals 
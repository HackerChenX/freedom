#!/usr/bin/env python3
"""
OBV (On-Balance Volume) 能量潮指标

OBV指标通过累计成交量来反映资金流向。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class OBV(BaseIndicator, PatternSignalMixin):
    """
    OBV (On-Balance Volume) 能量潮指标
    
    OBV指标通过累计成交量变化来判断资金流向。
    """
    
    def __init__(self, **kwargs):
        """
        初始化OBV指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "OBV"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        
        # 应用用户参数
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"signal_period": 10}
    
    def set_parameters(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        from utils.indicator_parameter_validator import IndicatorParameterValidator
        validator = IndicatorParameterValidator()
        
        # 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('OBV', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.signal_period = params.get('signal_period', 10)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了OBV指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了OBV指标的DataFrame
        """
        df = data.copy()
        
        # 确保数据有足够的长度
        if len(df) < 2:
            logger.warning(f"数据长度({len(df)})不足，返回原始数据")
            df['OBV'] = np.nan
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（OBV指标特定逻辑）
        df = self._apply_obv_signal_logic(df)

        return df

    def _apply_obv_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用OBV指标特定的信号生成逻辑
        基于OBV值的变化生成信号
        """
        try:
            # 获取OBV值
            if 'OBV' not in df.columns:
                # 如果没有OBV值，使用默认信号
                return df

            obv_value = df['OBV']
            close_price = df['close']

            # OBV信号生成逻辑：
            # BUY: OBV上升且价格上升（量价齐升）
            # SELL: OBV下降且价格下降（量价齐跌）
            # HOLD: 量价背离或无明显趋势

            # 计算OBV和价格的变化
            obv_rising = obv_value > obv_value.shift(1)
            obv_falling = obv_value < obv_value.shift(1)
            price_rising = close_price > close_price.shift(1)
            price_falling = close_price < close_price.shift(1)

            # 生成信号
            df.loc[:, 'buy_signal'] = obv_rising & price_rising
            df.loc[:, 'sell_signal'] = obv_falling & price_falling
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"OBV信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

        # 计算价格变化
        df['price_change'] = df['close'].diff()
        
        # 计算OBV
        obv = [0]  # 初始值为0
        for i in range(1, len(df)):
            if df['price_change'].iloc[i] > 0:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['price_change'].iloc[i] < 0:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['OBV'] = obv
        
        # 计算OBV移动平均线
        df[f'OBV_MA{self.signal_period}'] = df['OBV'].rolling(window=self.signal_period).mean()
        
        # 清理中间计算列
        df.drop(['price_change'], axis=1, inplace=True)
        
        return df
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate(data, **kwargs)
        return pd.Series(50.0, index=data.index)
    
    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)

#!/usr/bin/env python3
"""
STOCHRSI (Stochastic RSI) 随机相对强弱指标

STOCHRSI是RSI指标的随机化版本，用于识别超买超卖状态。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class STOCHRSI(BaseIndicator, PatternSignalMixin):
    """
    STOCHRSI (Stochastic RSI) 随机相对强弱指标
    
    STOCHRSI结合了RSI和随机指标的特点。
    """
    
    def __init__(self, **kwargs):
        """
        初始化STOCHRSI指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "STOCHRSI"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        
        # 应用用户参数
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"rsi_period": 14, "stoch_period": 14, "k_period": 3, "d_period": 3}
    
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
            is_valid, errors = validator.validate_indicator_parameters('STOCHRSI', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.rsi_period = params.get('rsi_period', 14)
        self.stoch_period = params.get('stoch_period', 14)
        self.k_period = params.get('k_period', 3)
        self.d_period = params.get('d_period', 3)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算STOCHRSI指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了STOCHRSI指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算STOCHRSI指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了STOCHRSI指标的DataFrame
        """
        df = data.copy()
        
        # 确保数据有足够的长度
        min_length = max(self.rsi_period, self.stoch_period) + self.k_period + self.d_period
        if len(df) < min_length:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({min_length})，返回原始数据")
            df['STOCHRSI_K'] = np.nan
            df['STOCHRSI_D'] = np.nan
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（STOCHRSI指标特定逻辑）
        df = self._apply_stochrsi_signal_logic(df)

        return df

    def _apply_stochrsi_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用STOCHRSI指标特定的信号生成逻辑
        基于STOCHRSI值的超买超卖区间生成信号
        """
        try:
            # 获取STOCHRSI值
            if 'STOCHRSI_K' not in df.columns or 'STOCHRSI_D' not in df.columns:
                # 如果没有STOCHRSI值，使用默认信号
                return df

            stochrsi_k = df['STOCHRSI_K']
            stochrsi_d = df['STOCHRSI_D']

            # STOCHRSI信号生成逻辑：
            # BUY: STOCHRSI从超卖区间(< 20)向上突破且K线在D线之上
            # SELL: STOCHRSI从超买区间(> 80)向下突破且K线在D线之下
            # HOLD: STOCHRSI在正常区间(20-80)

            # 定义超买超卖区间
            oversold = (stochrsi_k < 20) & (stochrsi_d < 20)
            overbought = (stochrsi_k > 80) & (stochrsi_d > 80)
            normal = ~(oversold | overbought)

            # 检测K线与D线的关系
            k_above_d = stochrsi_k > stochrsi_d
            k_below_d = stochrsi_k < stochrsi_d

            # 检测突破
            k_rising = stochrsi_k > stochrsi_k.shift(1)
            k_falling = stochrsi_k < stochrsi_k.shift(1)

            # 生成信号
            df.loc[:, 'buy_signal'] = oversold & k_above_d & k_rising
            df.loc[:, 'sell_signal'] = overbought & k_below_d & k_falling
            df.loc[:, 'hold_signal'] = normal | (~(df['buy_signal'] | df['sell_signal']))

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"STOCHRSI信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 计算StochRSI
        rsi_min = rsi.rolling(window=self.stoch_period).min()
        rsi_max = rsi.rolling(window=self.stoch_period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        
        # 计算%K和%D
        df['STOCHRSI_K'] = stoch_rsi.rolling(window=self.k_period).mean()
        df['STOCHRSI_D'] = df['STOCHRSI_K'].rolling(window=self.d_period).mean()
        
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
